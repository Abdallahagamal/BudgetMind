# BudgetMind Sprint 3 — Pipeline Testing Report (Load, Throughput, Latency)

**Track:** Backend / Infra
**Owner:** Abdallah Gamal
**Sprint Goal:** Make the classification layer reliable enough for integration with the rest of BudgetMind
**Scope:**
- Load testing (5 Points)
- Measure throughput (2 Points)
- Measure latency (2 Points)

---

## 1. Executive Summary

Sprint 2 (Eyad) proved the Service & Queue pipeline is **correct**: at-least-once delivery, zero false ACKs, and clean crash recovery. Sprint 3 asks a different question — is it **fast enough, and does it stay correct under volume**? This report covers both.

The system under test is the same one Sprint 2 verified: `RedisStreamPublisher/Consumer` → `ClassificationWorker` → `ClassificationService` (the finalized Sprint 2 `type`/`complexity`/`domain` classifiers) → JSONL persistence → `xack`. All measurements run against `fakeredis` (in-memory, no live Redis server) to keep the suite self-contained, matching the Sprint 2 testing convention — see [Section 6](#6-a-note-on-measurement-environment) for what that does and doesn't tell us.

### Summary of Results

| Category | Allocated Points | Scenarios | Passed | Status |
|---|:---:|:---:|:---:|:---:|
| **Load Testing** | 5 | 5 | 5 | **100% PASS** |
| **Throughput Measurement** | 2 | 3 | 3 | **DONE** |
| **Latency Measurement** | 2 | 2 | 2 | **DONE** |
| **TOTAL** | **9** | **10** | **10** | — |

### Headline Numbers

| Metric | Result |
|---|---|
| End-to-end throughput | **~1,540 tasks/sec** (steady from 100 to 2,000 tasks) |
| Classification-only throughput | **~4,060 tasks/sec** |
| End-to-end per-task latency (p50 / p99) | **0.83 ms / 1.07 ms** |
| Classification-only latency (p50 / p99) | **0.24 ms / 0.30 ms** |
| Queue + persistence overhead (p50) | **~0.59 ms/task** |
| Largest single-burst load tested | 1,000 tasks, 8-way concurrent publish, 600-task mixed-validity load |

---

## 2. Load Testing (5 Scenarios)

Located in [`Sprint3/tests/test_load.py`](file:///d:/BudgetMind/BudgetMind/Sprint3/tests/test_load.py). Each scenario asserts **correctness under volume** (no lost tasks, no false ACKs, no crashes) — throughput/latency numbers are covered separately in Sections 3–4.

### LOAD-1: Single Large Burst
- **Objective:** Verify the pipeline drains a large burst cleanly with no loss.
- **Workflow:** 1,000 tasks published in one burst, worker drains in batches of 50.
- **Result:** 1,000/1,000 processed in 20 cycles, 0 pending afterward.

### LOAD-2: Sustained Multi-Cycle Load
- **Objective:** Verify the queue doesn't accumulate a growing backlog under continuous publish/drain waves.
- **Workflow:** 20 cycles of 50 tasks each (1,000 total), draining between waves.
- **Result:** 1,000/1,000 persisted, max observed backlog stayed at 0 between waves, 0 pending at the end.

### LOAD-3: Concurrent Publishers, Single Worker
- **Objective:** Verify no lost or duplicated tasks when multiple producers write to the same stream concurrently.
- **Workflow:** 8 threads publishing 100 tasks each (800 total) against a shared `fakeredis` client, single worker draining.
- **Result:** 800/800 processed, 800 unique `task_id`s recovered (no duplicates, no drops), 0 pending afterward.

### LOAD-4: Small-Batch Backpressure Stress
- **Objective:** Verify the worker stays correct when forced into many small drain cycles rather than a few large ones.
- **Workflow:** 500 tasks, `batch_size=3` (deliberately tiny → 167 forced cycles).
- **Result:** 500/500 processed in exactly the theoretical minimum of 167 cycles, 0 pending.

### LOAD-5: Mixed Valid/Invalid Traffic Under Volume
- **Objective:** Verify malformed tasks don't corrupt or stall a large batch of otherwise-valid traffic (extends Sprint 2's FAIL-2 no-false-ACK guarantee to volume conditions).
- **Workflow:** 600 tasks, 10% (60) with wrong-dimension embeddings interleaved at the tail of the stream.
- **Result:** All 540 valid tasks processed and persisted; all 60 invalid tasks correctly left un-ACKed and pending (zero false ACKs); pipeline did not crash or stall on the invalid batch.

---

## 3. Throughput Measurement

Located in [`Sprint3/measure_throughput.py`](file:///d:/BudgetMind/BudgetMind/Sprint3/measure_throughput.py). Results: [`Sprint3/results/pipeline_throughput_results.json`](file:///d:/BudgetMind/BudgetMind/Sprint3/results/pipeline_throughput_results.json).

### THRU-1: Classification-Only Throughput
Isolates raw model-inference speed (`ClassificationService.classify()`), no queue or disk I/O.
- 2,000 tasks (50-task warmup excluded): **4,055.66 tasks/sec** (0.2466 ms/task average).

### THRU-2: End-to-End Pipeline Throughput
Full path: publish → dequeue → classify → persist → ACK.
- 1,000 tasks, `batch_size=50`: **1,541.33 tasks/sec**, 1,000/1,000 processed.

### THRU-3: Throughput vs. Load Size
Confirms throughput doesn't degrade as volume scales — no memory growth or slowdown under sustained load.

| Load Size | Throughput (tasks/sec) |
|---:|---:|
| 100 | 1,615.04 |
| 500 | 1,580.96 |
| 1,000 | 1,494.91 |
| 2,000 | 1,528.57 |

Throughput stays within a ~7% band across a 20x increase in load — no evidence of degradation at these volumes.

---

## 4. Latency Measurement

Located in [`Sprint3/measure_latency.py`](file:///d:/BudgetMind/BudgetMind/Sprint3/measure_latency.py). Results: [`Sprint3/results/pipeline_latency_results.json`](file:///d:/BudgetMind/BudgetMind/Sprint3/results/pipeline_latency_results.json).

### LAT-1: Classification-Only Latency
500 single-task `classify()` calls (50-task warmup excluded):

| p50 | p95 | p99 | max | mean |
|---:|---:|---:|---:|---:|
| 0.2403 ms | 0.2869 ms | 0.3025 ms | 0.4723 ms | 0.2481 ms |

### LAT-2: Per-Task Pipeline Latency
300 tasks, `batch_size=1` so each drain cycle's wall time is exactly one task's dequeue → classify → persist → ACK cost:

| p50 | p95 | p99 | max | mean |
|---:|---:|---:|---:|---:|
| 0.8284 ms | 0.9334 ms | 1.0693 ms | 1.2272 ms | 0.8364 ms |

**Queue + persistence overhead (p50): ≈0.59 ms/task** — the gap between LAT-1 and LAT-2, i.e. what Redis Streams read/ACK and the JSONL append add on top of raw model inference.

---

## 5. How to Run

```bash
# Everything in one pass (load + throughput + latency, formatted report)
python Sprint3/run_performance_tests.py
python Sprint3/run_performance_tests.py --verbose   # full tracebacks on failure

# Individually
python -m pytest Sprint3/tests/test_load.py -v
python Sprint3/tests/test_load.py                    # standalone, prints metrics
python Sprint3/measure_throughput.py                  # writes results/pipeline_throughput_results.json
python Sprint3/measure_latency.py                      # writes results/pipeline_latency_results.json
```

---

## 6. A Note on Measurement Environment

These numbers were captured against `fakeredis` (in-process, in-memory) on a single shared machine, not a live Redis server or production hardware. That's the right setup for proving **correctness under load** (Section 2) and for **relative** comparisons (THRU-3's scaling trend, the LAT-1 vs. LAT-2 overhead delta) — those conclusions hold regardless of what's underneath.

For **absolute** throughput/latency claims in the graduation report or dashboard, these figures should be re-captured against a real Redis instance and representative hardware, since network round-trips to an actual Redis server will add latency that `fakeredis` — with zero network cost — does not reflect. Everything in this report is scripted and reproducible ([Section 5](#5-how-to-run)), so re-running against live infrastructure is a config change (`BUDGETMIND_REDIS_HOST`/`BUDGETMIND_REDIS_PORT`), not a rewrite.

---

## 7. Deliverables & Code Changes Summary

- [`Sprint3/tests/test_load.py`](file:///d:/BudgetMind/BudgetMind/Sprint3/tests/test_load.py): 5 load scenarios (Burst, Sustained, Concurrent Publishers, Backpressure, Mixed Valid/Invalid).
- [`Sprint3/measure_throughput.py`](file:///d:/BudgetMind/BudgetMind/Sprint3/measure_throughput.py): Classification-only, end-to-end, and load-scaling throughput measurements.
- [`Sprint3/measure_latency.py`](file:///d:/BudgetMind/BudgetMind/Sprint3/measure_latency.py): Classification-only and per-task pipeline latency percentiles.
- [`Sprint3/run_performance_tests.py`](file:///d:/BudgetMind/BudgetMind/Sprint3/run_performance_tests.py): Unified runner producing the formatted summary in this report.
- [`Sprint3/results/pipeline_throughput_results.json`](file:///d:/BudgetMind/BudgetMind/Sprint3/results/pipeline_throughput_results.json), [`Sprint3/results/pipeline_latency_results.json`](file:///d:/BudgetMind/BudgetMind/Sprint3/results/pipeline_latency_results.json): Raw measurement output.
- [`Sprint3/docs/pipeline_performance_report.md`](file:///d:/BudgetMind/BudgetMind/Sprint3/docs/pipeline_performance_report.md): This document.
