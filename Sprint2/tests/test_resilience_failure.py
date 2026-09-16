from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import fakeredis
import numpy as np
import pytest

_CURRENT_DIR = Path(__file__).resolve().parent
_SPRINT2_DIR = _CURRENT_DIR.parent
_SRC_DIR = _SPRINT2_DIR / "src"

for p in (str(_SRC_DIR), str(_SPRINT2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from service.classification_service import ClassificationService
from streaming.redis_stream import (
    RedisStreamConfig,
    RedisStreamConsumer,
    RedisStreamPublisher,
)
from worker.classification_worker import ClassificationWorker


def generate_synthetic_embedding(dim: int = 384, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def test_failure_malformed_wire_payloads(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:fail:malformed",
        group_name="test:grp:malformed",
        consumer_name="test:worker:malformed",
        block_ms=300,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "malformed_output.jsonl"

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=output_file,
    )

    publisher.publish("T_VALID_1", generate_synthetic_embedding(seed=1))

    fake_client.xadd(config.stream_name, {"payload": "{broken_json_not_closing"})
    fake_client.xadd(config.stream_name, {"corrupt_header": "missing_payload"})
    fake_client.xadd(config.stream_name, {"payload": json.dumps({"only_one_key": "val"})})

    publisher.publish("T_VALID_2", generate_synthetic_embedding(seed=2))

    processed = worker.run_once()
    assert processed == 2

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 2
    assert records[0]["task_id"] == "T_VALID_1"
    assert records[1]["task_id"] == "T_VALID_2"

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    assert pending_count == 0
    consumer.close()


def test_failure_service_domain_invalid_embeddings_no_false_ack(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:fail:service",
        group_name="test:grp:service",
        consumer_name="test:worker:service",
        block_ms=300,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "service_fail_output.jsonl"

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=output_file,
    )

    publisher.publish("T_BAD_DIM", [0.1, 0.2, 0.3, 0.4, 0.5])

    nan_embedding = generate_synthetic_embedding(seed=50)
    nan_embedding[10] = float("nan")
    publisher.publish("T_BAD_NAN", nan_embedding)

    inf_embedding = generate_synthetic_embedding(seed=51)
    inf_embedding[20] = float("inf")
    publisher.publish("T_BAD_INF", inf_embedding)

    processed = worker.run_once()
    assert processed == 0

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 0

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    assert pending_count == 3
    consumer.close()


def test_failure_storage_io_no_false_ack(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:fail:storage",
        group_name="test:grp:storage",
        consumer_name="test:worker:storage",
        block_ms=300,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "storage_fail.jsonl"

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=output_file,
    )

    publisher.publish("T_IO_FAIL", generate_synthetic_embedding(seed=77))

    with patch.object(ClassificationWorker, "_write_profile", side_effect=OSError("Disk write failed: No space left on device")):
        processed = worker.run_once()
        assert processed == 0

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            records = [line for line in f if line.strip()]
        assert len(records) == 0

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    assert pending_count == 1
    consumer.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_failure_malformed_wire_payloads(p / "f1")
        test_failure_service_domain_invalid_embeddings_no_false_ack(p / "f2")
        test_failure_storage_io_no_false_ack(p / "f3")
