from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import fakeredis
import numpy as np
import pytest
import redis

_CURRENT_DIR = Path(__file__).resolve().parent
_SPRINT2_DIR = _CURRENT_DIR.parent
_SRC_DIR = _SPRINT2_DIR / "src"

for p in (str(_SRC_DIR), str(_SPRINT2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from service.classification_service import (
    ClassificationService,
    ClassificationServiceError,
)
from streaming.redis_stream import (
    RedisConnectionUnavailable,
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


def test_recovery_worker_crash_and_pel_reprocessing(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:recov:crash",
        group_name="test:grp:crash",
        consumer_name="worker_instance_1",
        block_ms=300,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer_1 = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "crash_recovery_output.jsonl"

    for i in range(3):
        publisher.publish(f"T_CRASH_{i}", generate_synthetic_embedding(seed=10 + i))

    delivered_msgs = list(consumer_1.read())
    assert len(delivered_msgs) == 3

    pending_before = consumer_1.pending()
    assert pending_before[0]["pending"] == 3

    consumer_1.close()

    consumer_restart = RedisStreamConsumer(config=config, client=fake_client)
    restarted_worker = ClassificationWorker(
        consumer=consumer_restart,
        service=service,
        output_path=output_file,
    )

    new_processed = restarted_worker.run_once()
    assert new_processed == 0

    recovered_count = restarted_worker.recover_pending()
    assert recovered_count == 3

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 3
    assert [r["task_id"] for r in records] == ["T_CRASH_0", "T_CRASH_1", "T_CRASH_2"]

    pending_after = consumer_restart.pending()
    pending_count = pending_after[0]["pending"] if pending_after else 0
    assert pending_count == 0

    consumer_restart.close()


def test_recovery_downstream_service_outage_and_self_healing(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:recov:outage",
        group_name="test:grp:outage",
        consumer_name="worker_resilience",
        block_ms=300,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    real_service = ClassificationService()
    output_file = tmp_path / "outage_recovery.jsonl"

    mock_service = MagicMock()
    mock_service.classify.side_effect = ClassificationServiceError("Service unavailable: Out of memory")

    worker = ClassificationWorker(
        consumer=consumer,
        service=mock_service,
        output_path=output_file,
    )

    publisher.publish("T_OUTAGE_1", generate_synthetic_embedding(seed=80))
    publisher.publish("T_OUTAGE_2", generate_synthetic_embedding(seed=81))

    processed_during_outage = worker.run_once()
    assert processed_during_outage == 0

    pending = consumer.pending()
    assert pending[0]["pending"] == 2

    mock_service.classify.side_effect = real_service.classify

    recovered = worker.recover_pending()
    assert recovered == 2

    with open(output_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    assert [r["task_id"] for r in records] == ["T_OUTAGE_1", "T_OUTAGE_2"]

    pending_after = consumer.pending()
    pending_count = pending_after[0]["pending"] if pending_after else 0
    assert pending_count == 0
    consumer.close()


def test_recovery_redis_connection_interruption_and_reconnection(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:recov:conn",
        group_name="test:grp:conn",
        consumer_name="worker_conn",
        block_ms=300,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "conn_recovery.jsonl"

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=output_file,
    )

    publisher.publish("T_CONN_1", generate_synthetic_embedding(seed=90))

    def failing_xreadgroup(*args, **kwargs):
        raise redis.exceptions.ConnectionError("Connection reset by peer")

    with patch.object(fake_client, "xreadgroup", side_effect=failing_xreadgroup):
        try:
            worker.run_once()
            connection_dropped_handled = False
        except RedisConnectionUnavailable:
            connection_dropped_handled = True

    assert connection_dropped_handled

    processed_after_restore = worker.run_once()
    assert processed_after_restore == 1

    publisher.publish("T_CONN_2", generate_synthetic_embedding(seed=91))
    processed_second = worker.run_once()
    assert processed_second == 1

    with open(output_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    assert len(records) == 2
    assert [r["task_id"] for r in records] == ["T_CONN_1", "T_CONN_2"]

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    assert pending_count == 0

    consumer.close()


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_recovery_worker_crash_and_pel_reprocessing(p / "r1")
        test_recovery_downstream_service_outage_and_self_healing(p / "r2")
        test_recovery_redis_connection_interruption_and_reconnection(p / "r3")
