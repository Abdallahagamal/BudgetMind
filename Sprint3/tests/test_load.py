
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

import fakeredis
import numpy as np

_CURRENT_DIR = Path(__file__).resolve().parent
_SPRINT3_DIR = _CURRENT_DIR.parent
_SPRINT2_DIR = _SPRINT3_DIR.parent / "Sprint2"
_SRC_DIR = _SPRINT2_DIR / "src"

for p in (str(_SRC_DIR), str(_SPRINT2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from service.classification_service import ClassificationService
from streaming.redis_stream import RedisStreamConfig, RedisStreamConsumer, RedisStreamPublisher
from worker.classification_worker import ClassificationWorker


def generate_synthetic_embedding(dim: int = 384, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _drain(worker: ClassificationWorker, expected: int, max_cycles: int = 200) -> tuple[int, int]:

    total_processed = 0
    cycles = 0
    while total_processed < expected and cycles < max_cycles:
        cycles += 1
        count = worker.run_once()
        if count == 0:
            break
        total_processed += count
    return total_processed, cycles


def _drain_fixed_cycles(worker: ClassificationWorker, total_entries: int, batch_size: int) -> tuple[int, int]:

    import math

    cycles = math.ceil(total_entries / batch_size)
    total_processed = 0
    for _ in range(cycles):
        total_processed += worker.run_once()
    return total_processed, cycles


# --------------------------------------------------------------------------- #
# LOAD-1: Single Large Burst
# --------------------------------------------------------------------------- #

def test_load_single_large_burst(tmp_path: Path, n_tasks: int = 1000) -> dict[str, Any]:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:load:burst",
        group_name="test:grp:burst",
        consumer_name="test:worker:burst",
        batch_size=50,
        block_ms=200,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "burst_profiles.jsonl"
    worker = ClassificationWorker(consumer=consumer, service=service, output_path=output_file)

    for i in range(n_tasks):
        publisher.publish(f"T_BURST_{i:05d}", generate_synthetic_embedding(seed=1000 + i))

    processed, cycles = _drain(worker, n_tasks)

    assert processed == n_tasks, f"expected {n_tasks} processed, got {processed}"
    with open(output_file, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]
    assert len(lines) == n_tasks, "output file record count mismatch"

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    assert pending_count == 0, f"expected 0 pending, got {pending_count}"
    consumer.close()

    return {"n_tasks": n_tasks, "processed": processed, "cycles": cycles, "pending_after": pending_count}


# --------------------------------------------------------------------------- #
# LOAD-2: Sustained Multi-Cycle Load
# --------------------------------------------------------------------------- #

def test_load_sustained_multi_cycle(tmp_path: Path, n_cycles: int = 20, per_cycle: int = 50) -> dict[str, Any]:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:load:sustained",
        group_name="test:grp:sustained",
        consumer_name="test:worker:sustained",
        batch_size=20,
        block_ms=200,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "sustained_profiles.jsonl"
    worker = ClassificationWorker(consumer=consumer, service=service, output_path=output_file)

    total_published = 0
    max_backlog = 0
    for cycle in range(n_cycles):
        for i in range(per_cycle):
            publisher.publish(f"T_SUS_{cycle:03d}_{i:03d}", generate_synthetic_embedding(seed=5000 + cycle * per_cycle + i))
            total_published += 1
        
        pending_before = consumer.pending()
        backlog = pending_before[0]["pending"] if pending_before else 0
        max_backlog = max(max_backlog, backlog)
        _drain(worker, expected=per_cycle, max_cycles=10)

    with open(output_file, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    consumer.close()

    assert len(lines) == total_published, f"expected {total_published} records, got {len(lines)}"
    assert pending_count == 0, f"backlog did not drain: {pending_count} still pending"

    return {
        "n_cycles": n_cycles,
        "per_cycle": per_cycle,
        "total_published": total_published,
        "total_persisted": len(lines),
        "max_observed_backlog": max_backlog,
        "pending_after": pending_count,
    }


# --------------------------------------------------------------------------- #
# LOAD-3: Concurrent Publishers, Single Worker
# --------------------------------------------------------------------------- #

def test_load_concurrent_publishers(tmp_path: Path, n_publishers: int = 8, per_publisher: int = 100) -> dict[str, Any]:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:load:concurrent",
        group_name="test:grp:concurrent",
        consumer_name="test:worker:concurrent",
        batch_size=32,
        block_ms=200,
    )
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "concurrent_profiles.jsonl"
    worker = ClassificationWorker(consumer=consumer, service=service, output_path=output_file)

    errors: list[Exception] = []

    def _publish_worker(publisher_id: int) -> None:
        try:
            pub = RedisStreamPublisher(config=config, client=fake_client)
            for i in range(per_publisher):
                pub.publish(f"T_CONC_{publisher_id:02d}_{i:04d}", generate_synthetic_embedding(seed=9000 + publisher_id * 1000 + i))
        except Exception as exc:  # pragma: no cover - surfaced via assertion below
            errors.append(exc)

    threads = [threading.Thread(target=_publish_worker, args=(pid,)) for pid in range(n_publishers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"publisher thread(s) raised: {errors}"

    total_tasks = n_publishers * per_publisher
    processed, cycles = _drain(worker, expected=total_tasks, max_cycles=500)

    with open(output_file, "r", encoding="utf-8") as f:
        records = [ln for ln in f if ln.strip()]


    import json as _json
    task_ids = {_json.loads(r)["task_id"] for r in records}

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    consumer.close()

    assert processed == total_tasks, f"expected {total_tasks} processed, got {processed}"
    assert len(task_ids) == total_tasks, "duplicate or missing task_ids under concurrent publish"
    assert pending_count == 0

    return {
        "n_publishers": n_publishers,
        "per_publisher": per_publisher,
        "total_tasks": total_tasks,
        "processed": processed,
        "unique_task_ids": len(task_ids),
        "cycles": cycles,
        "pending_after": pending_count,
    }


# --------------------------------------------------------------------------- #
# LOAD-4: Small-Batch Backpressure Stress
# --------------------------------------------------------------------------- #

def test_load_small_batch_backpressure(tmp_path: Path, n_tasks: int = 500) -> dict[str, Any]:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:load:backpressure",
        group_name="test:grp:backpressure",
        consumer_name="test:worker:backpressure",
        batch_size=3,  # deliberately tiny, to force many small cycles
        block_ms=100,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "backpressure_profiles.jsonl"
    worker = ClassificationWorker(consumer=consumer, service=service, output_path=output_file)

    for i in range(n_tasks):
        publisher.publish(f"T_BP_{i:04d}", generate_synthetic_embedding(seed=13000 + i))

    import math
    expected_min_cycles = math.ceil(n_tasks / config.batch_size)
    processed, cycles = _drain(worker, expected=n_tasks, max_cycles=expected_min_cycles + 5)

    with open(output_file, "r", encoding="utf-8") as f:
        lines = [ln for ln in f if ln.strip()]

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    consumer.close()

    assert processed == n_tasks
    assert len(lines) == n_tasks
    assert cycles == expected_min_cycles, f"expected exactly {expected_min_cycles} drain cycles at batch_size=3, got {cycles}"
    assert pending_count == 0

    return {
        "n_tasks": n_tasks,
        "batch_size": config.batch_size,
        "processed": processed,
        "cycles": cycles,
        "expected_min_cycles": expected_min_cycles,
        "pending_after": pending_count,
    }


# --------------------------------------------------------------------------- #
# LOAD-5: Mixed Valid/Invalid Traffic Under Volume
# --------------------------------------------------------------------------- #

def test_load_mixed_valid_invalid_traffic(tmp_path: Path, n_tasks: int = 600, invalid_ratio: float = 0.1) -> dict[str, Any]:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="test:load:mixed",
        group_name="test:grp:mixed",
        consumer_name="test:worker:mixed",
        batch_size=25,
        block_ms=200,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "mixed_profiles.jsonl"
    worker = ClassificationWorker(consumer=consumer, service=service, output_path=output_file)

    n_invalid = int(n_tasks * invalid_ratio)
    n_valid = n_tasks - n_invalid

    for i in range(n_valid):
        publisher.publish(f"T_MIX_VALID_{i:04d}", generate_synthetic_embedding(seed=20000 + i))
    for i in range(n_invalid):
        # Wrong-dimension embeddings: ClassificationService rejects these as
        # InvalidEmbeddingError; the worker must skip ACK, not crash the loop.
        publisher.publish(f"T_MIX_INVALID_{i:04d}", [0.1, 0.2, 0.3])

    processed, cycles = _drain_fixed_cycles(worker, total_entries=n_tasks, batch_size=config.batch_size)

    with open(output_file, "r", encoding="utf-8") as f:
        records = [ln for ln in f if ln.strip()]

    pending = consumer.pending()
    pending_count = pending[0]["pending"] if pending else 0
    consumer.close()

    assert processed == n_valid, f"expected {n_valid} valid tasks processed, got {processed}"
    assert len(records) == n_valid, "invalid tasks must not be persisted"
    assert pending_count == n_invalid, (
        f"expected exactly the {n_invalid} invalid tasks to remain pending "
        f"(no false ACK), got {pending_count}"
    )

    return {
        "n_tasks": n_tasks,
        "n_valid": n_valid,
        "n_invalid": n_invalid,
        "processed_valid": processed,
        "persisted_records": len(records),
        "pending_after": pending_count,
        "cycles": cycles,
    }


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        print("LOAD-1:", test_load_single_large_burst(p / "l1"))
        print("LOAD-2:", test_load_sustained_multi_cycle(p / "l2"))
        print("LOAD-3:", test_load_concurrent_publishers(p / "l3"))
        print("LOAD-4:", test_load_small_batch_backpressure(p / "l4"))
        print("LOAD-5:", test_load_mixed_valid_invalid_traffic(p / "l5"))
        print("\nAll load scenarios passed.")
