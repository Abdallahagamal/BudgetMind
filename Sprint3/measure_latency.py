from __future__ import annotations

import json
import statistics
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

RESULTS_PATH = _SPRINT3_DIR / "results" / "pipeline_latency_results.json"


def generate_synthetic_embedding(dim: int = 384, seed: int = 42) -> list[float]:
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    return {
        "p50_ms": round(statistics.median(ordered), 4),
        "p95_ms": round(ordered[int(len(ordered) * 0.95) - 1], 4),
        "p99_ms": round(ordered[int(len(ordered) * 0.99) - 1], 4),
        "max_ms": round(ordered[-1], 4),
        "min_ms": round(ordered[0], 4),
        "mean_ms": round(statistics.mean(ordered), 4),
        "n_samples": len(ordered),
    }


# --------------------------------------------------------------------------- #
# LAT-1: Classification-only latency
# --------------------------------------------------------------------------- #

def measure_classification_only_latency(n_tasks: int = 500, warmup: int = 50) -> dict[str, Any]:
    service = ClassificationService()
    embeddings = [generate_synthetic_embedding(seed=i) for i in range(n_tasks + warmup)]

    for i in range(warmup):
        service.classify(f"WARMUP_{i}", embeddings[i])

    samples_ms: list[float] = []
    for i in range(n_tasks):
        t0 = time.perf_counter()
        service.classify(f"LAT1_{i:05d}", embeddings[warmup + i])
        samples_ms.append((time.perf_counter() - t0) * 1000)

    return {"n_tasks": n_tasks, **_percentiles(samples_ms)}


# --------------------------------------------------------------------------- #
# LAT-2: Per-task pipeline latency (dequeue -> classify -> persist -> ACK)
# --------------------------------------------------------------------------- #

def measure_pipeline_per_task_latency(tmp_path: Path, n_tasks: int = 300) -> dict[str, Any]:
    fake_client = fakeredis.FakeRedis(decode_responses=True)
    config = RedisStreamConfig(
        stream_name="perf:lat:e2e",
        group_name="perf:grp:lat:e2e",
        consumer_name="perf:worker:lat:e2e",
        batch_size=1,  # one task per drain cycle, so each cycle's wall time
                       # is that single task's dequeue+classify+persist+ACK cost
        block_ms=200,
    )
    publisher = RedisStreamPublisher(config=config, client=fake_client)
    consumer = RedisStreamConsumer(config=config, client=fake_client)
    service = ClassificationService()
    output_file = tmp_path / "lat_e2e_profiles.jsonl"
    worker = ClassificationWorker(consumer=consumer, service=service, output_path=output_file)

    for i in range(n_tasks):
        publisher.publish(f"LAT2_{i:05d}", generate_synthetic_embedding(seed=11000 + i))

    samples_ms: list[float] = []
    for _ in range(n_tasks):
        t0 = time.perf_counter()
        count = worker.run_once()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if count == 0:
            break
        samples_ms.append(elapsed_ms)
    consumer.close()

    return {"n_tasks": len(samples_ms), **_percentiles(samples_ms)}


def main() -> None:
    import tempfile

    print("=" * 78)
    print("      BUDGETMIND SPRINT 3 - MEMBER 4: LATENCY MEASUREMENT      ")
    print("=" * 78)

    print("\n[LAT-1] Classification-only latency (model inference, no I/O)...")
    lat1 = measure_classification_only_latency()
    print(f"  p50={lat1['p50_ms']} ms  p95={lat1['p95_ms']} ms  p99={lat1['p99_ms']} ms  max={lat1['max_ms']} ms")

    print("\n[LAT-2] Per-task pipeline latency (dequeue -> classify -> persist -> ACK)...")
    with tempfile.TemporaryDirectory() as td:
        lat2 = measure_pipeline_per_task_latency(Path(td), n_tasks=300)
    print(f"  p50={lat2['p50_ms']} ms  p95={lat2['p95_ms']} ms  p99={lat2['p99_ms']} ms  max={lat2['max_ms']} ms")

    queue_overhead_p50 = round(lat2["p50_ms"] - lat1["p50_ms"], 4)

    results = {
        "classification_only": lat1,
        "pipeline_per_task": lat2,
        "estimated_queue_and_persistence_overhead_p50_ms": queue_overhead_p50,
    }

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nQueue + persistence overhead (p50): {queue_overhead_p50} ms/task")
    print(f"Results written to {RESULTS_PATH.relative_to(_SPRINT3_DIR.parent)}")
    print("=" * 78)


if __name__ == "__main__":
    main()
