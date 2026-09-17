from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, NamedTuple

_CURRENT_DIR = Path(__file__).resolve().parent
_TESTS_DIR = _CURRENT_DIR / "tests"

for p in (str(_TESTS_DIR), str(_CURRENT_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.test_load import (  # noqa: E402
    test_load_single_large_burst,
    test_load_sustained_multi_cycle,
    test_load_concurrent_publishers,
    test_load_small_batch_backpressure,
    test_load_mixed_valid_invalid_traffic,
)
from measure_throughput import (  # noqa: E402
    measure_classification_only_throughput,
    measure_end_to_end_throughput,
    measure_throughput_scaling,
)
from measure_latency import (  # noqa: E402
    measure_classification_only_latency,
    measure_pipeline_per_task_latency,
)


class LoadCase(NamedTuple):
    code: str
    name: str
    fn: Callable[[Path], dict]


LOAD_CASES = [
    LoadCase("LOAD-1", "Single Large Burst (1,000 tasks)", test_load_single_large_burst),
    LoadCase("LOAD-2", "Sustained Multi-Cycle Load (20 x 50 tasks)", test_load_sustained_multi_cycle),
    LoadCase("LOAD-3", "Concurrent Publishers, Single Worker (8 x 100 tasks)", test_load_concurrent_publishers),
    LoadCase("LOAD-4", "Small-Batch Backpressure Stress (batch_size=3, 500 tasks)", test_load_small_batch_backpressure),
    LoadCase("LOAD-5", "Mixed Valid/Invalid Traffic Under Volume (600 tasks, 10% invalid)", test_load_mixed_valid_invalid_traffic),
]

RESULTS_DIR = _CURRENT_DIR / "results"
LOAD_RESULTS_PATH = RESULTS_DIR / "pipeline_load_results.json"
THROUGHPUT_RESULTS_PATH = RESULTS_DIR / "pipeline_throughput_results.json"
LATENCY_RESULTS_PATH = RESULTS_DIR / "pipeline_latency_results.json"


def run_load_suite(verbose: bool) -> tuple[bool, list[dict]]:
    print("\n[Category: Load Testing]")
    results = []
    all_passed = True
    with tempfile.TemporaryDirectory() as td:
        base = Path(td)
        for case in LOAD_CASES:
            case_tmp = base / case.code
            case_tmp.mkdir(parents=True, exist_ok=True)
            t0 = time.perf_counter()
            try:
                metrics = case.fn(case_tmp)
                elapsed_ms = (time.perf_counter() - t0) * 1000
                print(f"  [{case.code}] {case.name:<58} ... OK ({elapsed_ms:7.1f} ms)")
                results.append({"code": case.code, "name": case.name, "status": "PASS", "elapsed_ms": elapsed_ms, "metrics": metrics})
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                all_passed = False
                print(f"  [{case.code}] {case.name:<58} ... FAILED ({elapsed_ms:7.1f} ms)")
                if verbose:
                    import traceback
                    traceback.print_exc()
                results.append({"code": case.code, "name": case.name, "status": "FAIL", "elapsed_ms": elapsed_ms, "error": str(exc)})
    return all_passed, results


def run_throughput_suite() -> dict:
    print("\n[Category: Throughput Measurement]")
    thru1 = measure_classification_only_throughput()
    print(f"  [THRU-1] Classification-only throughput ... {thru1['throughput_tasks_per_sec']:>9} tasks/sec")

    with tempfile.TemporaryDirectory() as td:
        thru2 = measure_end_to_end_throughput(Path(td), n_tasks=1000, batch_size=50)
    print(f"  [THRU-2] End-to-end pipeline throughput  ... {thru2['throughput_tasks_per_sec']:>9} tasks/sec")

    with tempfile.TemporaryDirectory() as td:
        thru3 = measure_throughput_scaling(Path(td))
    print(f"  [THRU-3] Throughput vs. load size        ... {thru3[0]['throughput_tasks_per_sec']:>9} - {thru3[-1]['throughput_tasks_per_sec']:>9} tasks/sec (100 -> 2000 tasks)")

    return {"classification_only": thru1, "end_to_end_pipeline": thru2, "throughput_vs_load_size": thru3}


def run_latency_suite() -> dict:
    print("\n[Category: Latency Measurement]")
    lat1 = measure_classification_only_latency()
    print(f"  [LAT-1] Classification-only latency ... p50={lat1['p50_ms']} ms  p99={lat1['p99_ms']} ms")

    with tempfile.TemporaryDirectory() as td:
        lat2 = measure_pipeline_per_task_latency(Path(td), n_tasks=300)
    print(f"  [LAT-2] Per-task pipeline latency    ... p50={lat2['p50_ms']} ms  p99={lat2['p99_ms']} ms")

    return {"classification_only": lat1, "pipeline_per_task": lat2}


def main() -> None:
    parser = argparse.ArgumentParser(description="BudgetMind Sprint 3 Performance Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose error output")
    args = parser.parse_args()

    print("=" * 78)
    print("   BUDGETMIND SPRINT 3 - MEMBER 4: PIPELINE TESTING SUITE   ")
    print("   Track: Backend / Infra  |  Owner: Member 4 (Service & Queue)   ")
    print("=" * 78)

    suite_start = time.perf_counter()
    load_passed, load_results = run_load_suite(args.verbose)
    throughput = run_throughput_suite()
    latency = run_latency_suite()
    suite_elapsed = time.perf_counter() - suite_start

    n_load_passed = sum(1 for r in load_results if r["status"] == "PASS")
    n_load_total = len(load_results)

    print("\n" + "=" * 78)
    print("                         RESULTS SUMMARY                          ")
    print("=" * 78)
    print(f"  Load Testing:   {n_load_passed}/{n_load_total} scenarios passed")
    print(f"  Throughput:     {throughput['end_to_end_pipeline']['throughput_tasks_per_sec']} tasks/sec (end-to-end)")
    print(f"  Latency:        p50={latency['pipeline_per_task']['p50_ms']} ms, p99={latency['pipeline_per_task']['p99_ms']} ms (end-to-end)")
    print(f"  Total Duration: {suite_elapsed:.2f}s")
    print("=" * 78)

    if load_passed:
        print(f"\n>>> ALL {n_load_total} LOAD SCENARIOS PASSED (100% SUCCESS) <<<\n")
    else:
        print(f"\n>>> {n_load_total - n_load_passed} LOAD SCENARIO(S) FAILED <<<\n")

    _write_results_json(load_results, throughput, latency)

    sys.exit(0 if load_passed else 1)


def _write_results_json(load_results: list[dict], throughput: dict, latency: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(LOAD_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({"scenarios": load_results}, f, indent=2)

    with open(THROUGHPUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(throughput, f, indent=2)

    with open(LATENCY_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(latency, f, indent=2)

    print(f"Results written to:")
    print(f"  {LOAD_RESULTS_PATH.relative_to(_CURRENT_DIR.parent)}")
    print(f"  {THROUGHPUT_RESULTS_PATH.relative_to(_CURRENT_DIR.parent)}")
    print(f"  {LATENCY_RESULTS_PATH.relative_to(_CURRENT_DIR.parent)}")


if __name__ == "__main__":
    main()