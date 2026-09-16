# Sprint 3 — TYPE Classifier (Esraa)

Covers: Model refinement, Test classification accuracy, contribution to the
shared performance sync, and edge cases handed off to M5.

---

## 1. Model refinement (5 pts)

Sprint 2's own `docs/phase3_model_selection.md` already named the refinement
to do here: TYPE has the worst calibration (ECE 0.338) of the three
dimensions, and the doc explicitly deferred fixing it — *"wrap the final
model in `CalibratedClassifierCV`. That is a Sprint 3 concern."* Algorithm
choice itself was **not** revisited: Sprint 2's error analysis already ran a
14-model sweep and found nothing beat `LogisticRegression`, and Summarization's
weakness is a data-coverage limit (single-source news data), not something a
different algorithm fixes.

### What was tried
`CalibratedClassifierCV` wrapping `LogisticRegression(C=1.0, class_weight="balanced")`,
sigmoid (Platt) and isotonic methods, cv=3 and cv=5 — 4 candidates, selected on
**validation only** (test untouched until the final report).

### A finding before the selection: isotonic is risky here
Summarization has only 29 examples total (~24 available for calibration after
holding out test) — squarely in the range where isotonic regression is known
to overfit. So sigmoid was preferred **a priori**, before any results were
compared, not chosen after peeking at outcomes. The data backs this up:
isotonic's validation ECE looked best (0.037–0.040) but cost real test-set
accuracy (Summarization F1 dropped as low as 0.600, vs sigmoid's 0.667–0.889)
— exactly the overfitting pattern the small-sample concern predicted.

### Result: `sigmoid, cv=3` selected

| | Sprint 2 baseline | Sprint 3 refined |
|---|---|---|
| Test accuracy | 0.964 | 0.964 |
| Test Macro-F1 | 0.962 | 0.950 |
| **Test ECE** | 0.308 | **0.117** (−62%) |
| Inference (83 rows) | 0.18ms | 1.42ms |

**The real payoff is the margin gate**, which is what calibration is actually
for (Decision 2's cache admission):

| Margin threshold | Baseline: admitted / accuracy | Refined: admitted / accuracy |
|---|---|---|
| ≥ 0.3 | 70/83 &nbsp;/&nbsp; 1.000 | **78/83** &nbsp;/&nbsp; 1.000 |
| ≥ 0.5 | 56/83 &nbsp;/&nbsp; 1.000 | **76/83** &nbsp;/&nbsp; 1.000 |

At the same 100% reliability, the refined model lets substantially more tasks
through the cache gate — 22 more at the 0.5 threshold. That's the concrete
benefit of fixing calibration: better-calibrated probabilities make the
margin a more trustworthy signal, so the gate doesn't have to be as
conservative to hit the same accuracy bar.

**The cost:** 0.012 Macro-F1, concentrated in Summarization (already a known,
documented weak class — this refinement doesn't fix that, it just changes
how honestly the model reports its uncertainty about it).

### Not deployed automatically
Saved as `Sprint3/models/type_classifier_refined.pkl`, **separate from**
`Sprint2/models/type_classifier_baseline.pkl`. Swapping the production
baseline is a team call, same as the original Phase 3 selection — bring this
comparison to the shared performance sync rather than treating it as decided.

Full numeric detail (all 4 calibration candidates, per-class breakdown, full
margin-gate table): `Sprint3/results/type_refinement_report.json`.

---

## 2. Test classification accuracy (3 pts)

Ran a regression check confirming the classifier performs as documented in
Sprint 2, in this environment, before relying on it further this sprint:

- Embeddings still 384-dim, L2-normalized, test set still 83 rows ✓
- Sprint 2 baseline reproduces its recorded test accuracy (0.964) and
  Macro-F1 (0.962) within tolerance — **no drift** ✓
- Interface contract (`predict_with_margin()`) still returns valid
  `{label, confidence, margin}` ✓
- Refined model evaluated the same way for comparison (see above)

**All checks passed.** Full detail: `Sprint3/results/type_accuracy_test.json`.

---

## 3. Shared "Measure classification performance" sync

Standardized summary block (same schema M2/M3 will use for Complexity/Domain,
built on the team's existing `evaluation.py`, not a one-off format) —
`Sprint3/results/type_performance_sync.json`. Ready to merge into one shared
report once M2 and M3 produce their equivalents.

---

## 4. Extra: edge cases handed off to M5

`Sprint3/docs/type_edge_cases_for_e2e.md` — the 2 known test-set errors (both
low-margin, useful for testing the gate catches them), the Summarization weak
spot with 3 fresh example prompts, the 3 TYPE labels with zero cross-source
evidence, margin-boundary test cases, and schema assertions to run on every
prediction. Told M5 to keep testing against the Sprint 2 baseline, not the
refined model, since the refined one isn't a production decision yet.

---

## Files produced this sprint

```
Sprint3/
├── src/
│   ├── calibrate_type.py          model refinement (calibration search + comparison)
│   ├── test_accuracy_type.py      regression test vs Sprint 2 recorded numbers
│   └── performance_sync_type.py   shared-sync summary generator
├── models/
│   └── type_classifier_refined.pkl
├── results/
│   ├── type_refinement_report.json
│   ├── type_accuracy_test.json
│   └── type_performance_sync.json
└── docs/
    └── type_edge_cases_for_e2e.md
```

## What's still open

- Team decision at the performance sync: adopt the calibrated model as the
  new baseline, or keep Sprint 2's — the accuracy/calibration tradeoff above
  is the case for it, but that's not mine to decide alone.
- "Contribute to shared classification performance sync" is only my slice —
  waiting on M2 (Complexity) and M3 (Domain) to produce their equivalents
  before the three can be merged into one report.
- Nothing here addresses the Summarization weak class itself or the 3
  zero-evidence TYPE labels — those remain data-coverage limitations per
  Sprint 2's error analysis, flagged again for M5's test suite rather than
  re-litigated here.
