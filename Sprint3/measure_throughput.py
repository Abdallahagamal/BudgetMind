from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import fakeredis
import numpy as np

_CURRENT_DIR = Path(__file__).resolve().parent
_SPRINT3_DIR = _CURRENT_DIR
_SPRINT2_DIR = _SPRINT3_DIR.parent / "Sprint2"
_SRC_DIR = _SPRINT2_DIR / "src"

for p in (str(_SRC_DIR), str(_SPRINT2_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from service.classification_service import ClassificationService
from streaming.redis_stream import RedisStreamConfig, RedisStreamConsumer, RedisStreamPublisher
from worker.classification_worker import ClassificationWorker

RESULTS_PATH = _SPRINT3_DIR / "results" / "pipeline_throughput_results.json"


def generate_synthetic_embedding(dim: int = 384, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


# --------------------------------------------------------------------------- #
# THRU-1: Classification-only throughput
# --------------------------------------------------------------------------- #

def measure_classification_only_throughput(n_tasks: int = 2000, warmup: int = 50) -> dict[str, Any]:
    service = ClassificationService()
    embeddings = [generate_synthetic_embedding(seed=i) for i in range(n_tasks + warmup)]


    for i in range(warmup):
        service.classify(f"WARMUP_{i}", embeddings[i])

    start = time.perf_counter()
    for i in range(n_tasks):
        service.classify(f"THRU1_{i:05d}", embeddings[warmup + i])
    elapsed = time.perf_counter() - start

    throughput = n_tasks / elapsed if elapsed > 0 else float("inf")
    return {
        "n_tasks": n_tasks,
        "warmup_tasks": warmup,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_tasks_per_sec": round(throughput, 2),
        "avg_ms_per_task": round((elapsed / n_tasks) * 1000, 4),
    }


# --------------------------------------------------------------------------- #
# THRU-2: End-to-end pipeline throughput
# --------------------------------------------------------------------------- #

def measure_end_to_end_throughput(tmp_path: Path, n_tasks: int = 1000, batch_size: int = 50) -> dict[str, Any]:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name=f"perf:thru:e2e:{n_tasks}",
        group_name="perf:grp:thru:e2e",
        consumer_name="perf:worker:thru:e2e",
        batch_size=batch_size,
        block_ms=200,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "thru_e2e_profiles.jsonl"
    worker = ClassificationWorker(consumer=consumer, service=service, output_path=output_file)

    for i in range(n_tasks):
        publisher.publish(f"THRU2_{i:05d}", generate_synthetic_embedding(seed=7000 + i))

    start = time.perf_counter()
    total_processed = 0
    while total_processed < n_tasks:
        count = worker.run_once()
        if count == 0:
            break
        total_processed += count
    elapsed = time.perf_counter() - start
    consumer.close()

    throughput = total_processed / elapsed if elapsed > 0 else float("inf")
    return {
        "n_tasks": n_tasks,
        "batch_size": batch_size,
        "processed": total_processed,
        "elapsed_seconds": round(elapsed, 4),
        "throughput_tasks_per_sec": round(throughput, 2),
    }


# --------------------------------------------------------------------------- #
# THRU-3: Throughput vs. load size
# --------------------------------------------------------------------------- #

def measure_throughput_scaling(tmp_path: Path, sizes: list[int] = [100, 500, 1000, 2000]) -> list[dict[str, Any]]:
    rows = []
    for size in sizes:
        result = measure_end_to_end_throughput(tmp_path / f"scale_{size}", n_tasks=size, batch_size=50)
        rows.append(result)
    return rows


def main() -> None:
    import tempfile

    print("=" * 78)
    print("      BUDGETMIND SPRINT 3 - MEMBER 4: THROUGHPUT MEASUREMENT      ")
    print("=" * 78)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        print("\n[THRU-1] Classification-only throughput (model inference, no I/O)...")
        thru1 = measure_classification_only_throughput()
        print(f"  -> {thru1['throughput_tasks_per_sec']} tasks/sec  ({thru1['avg_ms_per_task']} ms/task avg)")

        print("\n[THRU-2] End-to-end pipeline throughput (publish -> ... -> ACK)...")
        thru2 = measure_end_to_end_throughput(tmp / "thru2", n_tasks=1000, batch_size=50)
        print(f"  -> {thru2['throughput_tasks_per_sec']} tasks/sec  ({thru2['processed']}/{thru2['n_tasks']} processed)")

        print("\n[THRU-3] Throughput vs. load size (100 / 500 / 1000 / 2000 tasks)...")
        thru3 = measure_throughput_scaling(tmp / "thru3")
        for row in thru3:
            print(f"  n={row['n_tasks']:>5}  ->  {row['throughput_tasks_per_sec']:>8} tasks/sec")

        results = {
            "classification_only": thru1,
            "end_to_end_pipeline": thru2,
            "throughput_vs_load_size": thru3,
        }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {RESULTS_PATH.relative_to(_SPRINT3_DIR.parent)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
