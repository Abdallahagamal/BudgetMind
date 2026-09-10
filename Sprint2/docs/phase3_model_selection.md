# Phase 3 — Baseline Model Selection

Decision record for the Type / Complexity / Domain classifiers.

**Decision: `LogisticRegression(C=1.0, class_weight="balanced")` for all three
dimensions.**

---

## 1. Comparison

Selection metric is GROUPED (leave-one-source-out) macro-F1. RANDOM is reported
alongside because the gap between them is a thesis result, not because it
informs the choice.

| Dimension | Majority | Model | RANDOM | **GROUPED** | Nested | ECE | Inference |
| :---- | ----- | :---- | ----- | ----- | ----- | ----- | ----- |
| **Type** | 0.050 | **LogisticRegression** | 0.961 | **0.762** | **0.715** | 0.338 | 0.15 ms |
| | | RandomForest | 0.951 | 0.493 | — | 0.468 | 13.4 ms |
| | | SVM | 0.986 | 0.617 | — | 0.100 | 3.2 ms |
| **Complexity** | 0.238 | **LogisticRegression** | 0.894 | **0.702** | **0.679** | 0.137 | 0.15 ms |
| | | RandomForest | 0.797 | 0.312 | — | 0.161 | 13.4 ms |
| | | SVM | 0.904 | 0.640 | — | 0.021 | 3.2 ms |
| **Domain** | 0.151 | **LogisticRegression** | 0.969 | **0.491** | **0.477** | 0.222 | 0.15 ms |
| | | RandomForest | 0.886 | 0.186 | — | 0.315 | 13.4 ms |
| | | SVM | 0.992 | 0.369 | — | 0.026 | 3.2 ms |

*Inference is for the full 83-row test set.*

## 2. Why LogisticRegression

**It wins the selection metric on all three dimensions**, by a margin that is
not close: +0.145 over SVM on Type, +0.062 on Complexity, +0.122 on Domain.

**RandomForest fails and should be reported as failing.** On Domain it scores
0.186 grouped against a 0.151 majority baseline — barely distinguishable from
predicting the most common class. Tree ensembles split on individual dimensions
of a 384-dim dense embedding, where no single dimension is meaningful; this is
the expected outcome and it is worth one sentence in the thesis.

**SVM is a genuine runner-up but loses on every secondary criterion**: 21× slower
inference, and its probabilities come from an internal Platt-scaling CV, which
makes `.predict()` and `predict_proba().argmax()` capable of disagreeing — an
avoidable failure mode given the cache reads the same distribution.

**Calibration is the one criterion LogisticRegression loses on, and it is worth
being explicit about.** At C = 1.0 its ECE is 0.338 / 0.137 / 0.222, against
SVM's 0.100 / 0.021 / 0.026. Increasing regularization improved generalization
and hurt absolute calibration — a real trade-off, not a rounding artifact.

We accept it because **the cache gate depends on the ranking of margins, not on
their absolute values.** Measured end-to-end on the test set with the final
C = 1.0 models, gating on `min_margin` still behaves exactly as Decision 2
intends, and in fact better than the C = 10 models did:

| Gate | Admitted | All three labels correct |
| :---- | ----- | ----- |
| ungated | 83/83 | 0.843 |
| min margin ≥ 0.3 | 59/83 | 0.932 |
| min margin ≥ 0.5 | 47/83 | **0.979** |

If a later component needs probabilities to mean what they say in absolute terms
— rather than merely to rank correctly — wrap the final model in
`CalibratedClassifierCV`. That is a Sprint 3 concern; nothing in the committed
architecture needs it today.

## 3. Why C = 1.0

Chosen by **nested leave-one-source-out CV** (`Sprint2/src/nested_cv.py`), where
C is selected inside each fold using only the training sources, so the held-out
source never influences it.

| Dimension | C chosen per fold | Modal C | Nested macro-F1 |
| :---- | :---- | ----- | ----- |
| Type | 0.1×2, 1.0×3, 10.0×3, 100.0×2 | 1.0 | 0.715 |
| Complexity | 0.1×2, 1.0×5, 10.0×3, 100.0×3 | 1.0 | 0.679 |
| Domain | 0.1×2, 1.0×2, 10.0×4, 100.0×5 | 100.0 | 0.477 |

**C is not a sensitive parameter here.** No value wins a majority of folds on any
dimension, and across the wider sweep the spread over the whole grid was under
0.03 macro-F1. Given near-ties, C = 1.0 is used uniformly because it is the
modal choice on two of three dimensions, it is the more regularized option at
n = 377, and one value across three classifiers is simpler to reproduce and
defend than three tuned values that the evidence does not support. Domain's
modal C = 100.0 is recorded here but not adopted: the difference is 0.005
macro-F1, well inside fold-to-fold noise.

### Grouped vs nested figures

Choosing C on the grouped leaderboard and then reporting that same grouped score
selects on the reported metric. Nested CV quantifies the resulting optimism:
**Type 0.762 → 0.715, Complexity 0.702 → 0.679, Domain 0.491 → 0.477.** The bias
is small and does not change the ranking of candidates, but the nested figures
are the defensible ones and are what the thesis should quote.

## 4. What was accepted, with caveats

All three classifiers are accepted as Sprint 2 baselines. Three caveats carry
into the thesis and into Sprint 3:

- **Domain / Mathematics & Quantitative is effectively unlearned** (grouped
  F1 = 0.000; 39 of 40 rows come from one source). Do not route on this field
  for math tasks — TYPE's Mathematical Reasoning covers the same ground and is
  better supported.
- **The Margin Sampling gate has a known hole on Complexity**: 5 of 8 test errors
  carry margin ≥ 0.5 and would be admitted to the cache while wrong.
- **Three TYPE labels** (Translation, Mathematical Reasoning, Creative &
  Open-Ended Generation) have no cross-source evidence at all.

See `error_analysis.md` for the full analysis behind each.

## 5. Artifacts

| File | Contents |
| :---- | :---- |
| `Sprint2/models/{dimension}_classifier_baseline.pkl` | Final models, fit on train + validation |
| `Sprint2/results/{dimension}_results.json` | Both protocols, all candidates, per-class, errors |
| `Sprint2/results/{dimension}_nested_cv.json` | Nested CV folds and selected C |

Reproduce with:

```bash
python Sprint2/src/train_classifier.py --dimension all
python Sprint2/src/nested_cv.py --dimension all
```
