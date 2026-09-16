from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import fakeredis
import numpy as np
import pytest

_CURRENT_DIR = Path(__file__).resolve().parent
_SPRINT2_DIR = _CURRENT_DIR.parent
_SRC_DIR = _SPRINT2_DIR / "src"
_REPO_ROOT = _SPRINT2_DIR.parent

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


def test_e2e_full_pipeline_golden_path(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:e2e:golden",
        group_name="test:grp:golden",
        consumer_name="test:worker:golden",
        block_ms=500,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "golden_profiles.jsonl"

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=output_file,
    )

    tasks = [
        (f"T_GOLDEN_{i:03d}", generate_synthetic_embedding(seed=100 + i))
        for i in range(5)
    ]
    published_ids = []
    for task_id, emb in tasks:
        msg_id = publisher.publish(task_id, emb)
        published_ids.append(msg_id)

    assert len(published_ids) == 5

    processed = worker.run_once()
    assert processed == 5

    assert output_file.exists()
    with open(output_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) == 5
    saved_ids = [r["task_id"] for r in records]
    expected_ids = [t[0] for t in tasks]
    assert saved_ids == expected_ids

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    assert pending_count == 0
    consumer.close()


def test_e2e_multi_batch_queue_draining(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    batch_size = 7
    total_tasks = 23
    config = RedisStreamConfig(
        stream_name="test:e2e:multibatch",
        group_name="test:grp:multibatch",
        consumer_name="test:worker:multibatch",
        batch_size=batch_size,
        block_ms=300,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "multibatch_profiles.jsonl"

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=output_file,
    )

    for i in range(total_tasks):
        publisher.publish(f"T_MB_{i:03d}", generate_synthetic_embedding(seed=200 + i))

    total_processed = 0
    cycles = 0
    while total_processed < total_tasks and cycles < 10:
        cycles += 1
        count = worker.run_once()
        if count == 0:
            break
        total_processed += count

    assert total_processed == total_tasks
    assert cycles == 4

    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert len(lines) == total_tasks

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    assert pending_count == 0
    consumer.close()


def test_e2e_contract_conformance_and_margin_sampling(tmp_path: Path) -> None:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:e2e:contract",
        group_name="test:grp:contract",
        consumer_name="test:worker:contract",
        block_ms=500,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "contract_profiles.jsonl"

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=output_file,
    )

    for i in range(10):
        publisher.publish(f"T_CONTRACT_{i:03d}", generate_synthetic_embedding(seed=300 + i))

    processed = worker.run_once()
    assert processed == 10

    with open(output_file, "r", encoding="utf-8") as f:
        profiles = [json.loads(line) for line in f if line.strip()]

    required_contract_fields = {
        "task_id",
        "taxonomy_version",
        "type",
        "complexity",
        "domain",
        "type_confidence",
        "complexity_confidence",
        "domain_confidence",
        "type_margin",
        "complexity_margin",
        "domain_margin",
        "embedding_ref",
        "classified_at",
        "from_cache",
    }

    for p in profiles:
        assert required_contract_fields.issubset(p.keys())

        assert p["taxonomy_version"] == "1.0"
        assert p["type"] in service._models.type_model.classes_
        assert p["complexity"] in service._models.complexity_model.classes_
        assert p["domain"] in service._models.domain_model.classes_
        assert p["from_cache"] is False
        assert p["embedding_ref"] == f"emb:{p['task_id']}"

        for dim in ("type", "complexity", "domain"):
            conf = p[f"{dim}_confidence"]
            margin = p[f"{dim}_margin"]
            assert 0.0 <= conf <= 1.0
            assert 0.0 <= margin <= 1.0
            assert margin <= conf + 1e-6

    consumer.close()


def test_e2e_storage_append_integrity(tmp_path: Path) -> None:
    output_file = tmp_path / "append_test.jsonl"

    initial_records = [
        {"task_id": f"T_INIT_{i}", "taxonomy_version": "1.0", "type": "Factual Q&A"}
        for i in range(3)
    ]
    with open(output_file, "w", encoding="utf-8") as f:
        for rec in initial_records:
            f.write(json.dumps(rec) + "\n")

    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:e2e:append",
        group_name="test:grp:append",
        consumer_name="test:worker:append",
        block_ms=500,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()

    worker = ClassificationWorker(
        consumer=consumer,
        service=service,
        output_path=output_file,
    )

    for i in range(4):
        publisher.publish(f"T_NEW_{i}", generate_synthetic_embedding(seed=400 + i))

    processed = worker.run_once()
    assert processed == 4

    with open(output_file, "r", encoding="utf-8") as f:
        all_records = [json.loads(line) for line in f if line.strip()]

    assert len(all_records) == 7
    for i in range(3):
        assert all_records[i]["task_id"] == f"T_INIT_{i}"
    for i in range(4):
        assert all_records[3 + i]["task_id"] == f"T_NEW_{i}"

    consumer.close()


def test_e2e_pipeline_cli_execution(tmp_path: Path) -> None:
    output_file = tmp_path / "cli_output.jsonl"
    run_script = _SPRINT2_DIR / "run_pipeline.py"

    embeddings_dir = _REPO_ROOT / "Sprint1" / "embeddings_output"
    if not embeddings_dir.exists():
        embeddings_dir = _REPO_ROOT / "Sprint1" / "budgetmind_embeddings" / "embeddings_output_testing"

    cmd = [
        sys.executable,
        str(run_script),
        "--embeddings-dir", str(embeddings_dir),
        "--splits", "test",
        "--output", str(output_file),
        "--clear-output",
        "--fake",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_SPRINT2_DIR))
    assert result.returncode == 0

    assert "BUDGETMIND SPRINT 2 PIPELINE SUMMARY" in result.stdout
    assert "Total tasks classified:" in result.stdout
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    assert len(records) > 0
    assert records[0]["taxonomy_version"] == "1.0"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_e2e_full_pipeline_golden_path(p / "t1")
        test_e2e_multi_batch_queue_draining(p / "t2")
        test_e2e_contract_conformance_and_margin_sampling(p / "t3")
        test_e2e_storage_append_integrity(p / "t4")
        test_e2e_pipeline_cli_execution(p / "t5")
