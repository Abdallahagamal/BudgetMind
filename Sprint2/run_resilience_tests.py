from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, NamedTuple

_CURRENT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _CURRENT_DIR / "src"
_TESTS_DIR = _CURRENT_DIR / "tests"

for p in (str(_SRC_DIR), str(_CURRENT_DIR), str(_TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from tests.test_resilience_end_to_end import (
    test_e2e_full_pipeline_golden_path,
    test_e2e_multi_batch_queue_draining,
    test_e2e_contract_conformance_and_margin_sampling,
    test_e2e_storage_append_integrity,
    test_e2e_pipeline_cli_execution,
)
from tests.test_resilience_failure import (
    test_failure_malformed_wire_payloads,
    test_failure_service_domain_invalid_embeddings_no_false_ack,
    test_failure_storage_io_no_false_ack,
)
from tests.test_resilience_recovery import (
    test_recovery_worker_crash_and_pel_reprocessing,
    test_recovery_downstream_service_outage_and_self_healing,
    test_recovery_redis_connection_interruption_and_reconnection,
)


class TestCase(NamedTuple):
    category: str
    code: str
    name: str
    points: int
    fn: Callable[[Path], None]


TEST_CASES = [
    TestCase("End-to-End Functional", "E2E-1", "Full Pipeline Golden Path (Ingest -> Classify -> Save -> ACK)", 1, test_e2e_full_pipeline_golden_path),
    TestCase("End-to-End Functional", "E2E-2", "Multi-Batch Streaming & Queue Draining", 1, test_e2e_multi_batch_queue_draining),
    TestCase("End-to-End Functional", "E2E-3", "TaskProfile Contract & Margin Sampling Invariants", 1, test_e2e_contract_conformance_and_margin_sampling),
    TestCase("End-to-End Functional", "E2E-4", "Storage Append & File Format Integrity", 1, test_e2e_storage_append_integrity),
    TestCase("End-to-End Functional", "E2E-5", "Pipeline CLI Orchestration (run_pipeline.py)", 1, test_e2e_pipeline_cli_execution),

    TestCase("Failure Scenarios", "FAIL-1", "Malformed Wire Payloads & Poison Pill Isolation", 1, test_failure_malformed_wire_payloads),
    TestCase("Failure Scenarios", "FAIL-2", "Classification Domain Failure & PEL Preservation (No False ACK)", 1, test_failure_service_domain_invalid_embeddings_no_false_ack),
    TestCase("Failure Scenarios", "FAIL-3", "Storage I/O Write Failure & At-Least-Once ACK Abort", 1, test_failure_storage_io_no_false_ack),

    TestCase("Recovery Scenarios", "RECOV-1", "Worker Crash & Pending Entries List (PEL) Recovery", 1, test_recovery_worker_crash_and_pel_reprocessing),
    TestCase("Recovery Scenarios", "RECOV-2", "Downstream Service Temporary Outage & Self-Healing", 1, test_recovery_downstream_service_outage_and_self_healing),
    TestCase("Recovery Scenarios", "RECOV-3", "Redis Connection Interruption & Resume", 1, test_recovery_redis_connection_interruption_and_reconnection),
]


def run_all_tests(verbose: bool = False) -> bool:
    print("\n" + "=" * 78)
    print("      BUDGETMIND SPRINT 2 - MEMBER 5: RESILIENCE TESTING SUITE      ")
    print("      Owner: Member 5 (Worker & Integration Ownership)               ")
    print("=" * 78)

    total_tests = len(TEST_CASES)
    passed_tests = 0
    failed_tests = 0
    current_category = None
    suite_start = time.perf_counter()

    results: list[dict] = []

    with tempfile.TemporaryDirectory() as td:
        base_tmp = Path(td)

        for tc in TEST_CASES:
            if tc.category != current_category:
                current_category = tc.category
                print(f"\n[Category: {current_category}]")

            test_tmp = base_tmp / tc.code
            test_tmp.mkdir(parents=True, exist_ok=True)

            t0 = time.perf_counter()
            try:
                tc.fn(test_tmp)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                passed_tests += 1
                status = "PASS"
                error_msg = ""
                print(f"  [{tc.code}] {tc.name:<55} ... OK ({elapsed_ms:6.1f} ms)")
            except Exception as exc:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                failed_tests += 1
                status = "FAIL"
                error_msg = str(exc)
                print(f"  [{tc.code}] {tc.name:<55} ... FAILED ({elapsed_ms:6.1f} ms)")
                if verbose:
                    import traceback
                    traceback.print_exc()

            results.append({
                "category": tc.category,
                "code": tc.code,
                "name": tc.name,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "error": error_msg,
            })

    suite_elapsed = time.perf_counter() - suite_start

    print("\n" + "=" * 78)
    print("                            TEST RESULTS SUMMARY                          ")
    print("=" * 78)
    print(f"  {'Code':<8} {'Category':<24} {'Status':<8} {'Time (ms)':<10} {'Scenario Description'}")
    print("  " + "-" * 74)
    for r in results:
        status_str = f"\033[92m{r['status']}\033[0m" if r["status"] == "PASS" else f"\033[91m{r['status']}\033[0m"
        print(f"  {r['code']:<8} {r['category']:<24} {r['status']:<8} {r['elapsed_ms']:>8.1f} ms  {r['name'][:30]}")
    print("  " + "-" * 74)

    print(f"\n  Total Scenarios Executed: {total_tests}")
    print(f"  Passed:                   {passed_tests}")
    print(f"  Failed:                   {failed_tests}")
    print(f"  Total Duration:           {suite_elapsed:.2f}s")
    print("=" * 78)

    if failed_tests == 0:
        print("\n>>> ALL 11 RESILIENCE & INTEGRATION TESTS PASSED (100% SUCCESS) <<<\n")
        return True
    else:
        print(f"\n>>> {failed_tests} TEST(S) FAILED <<<\n")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BudgetMind Resilience Test Runner")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose error output")
    args = parser.parse_args()

    success = run_all_tests(verbose=args.verbose)
    sys.exit(0 if success else 1)
