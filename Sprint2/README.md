# Sprint 2 — Task Classifier

Training, evaluation, and the Task Profile contract for the Type / Complexity /
Domain classifiers.

Sprint 1 artifacts are **read-only** to this directory. Nothing here modifies the
taxonomy, the labels, the splits, or the embeddings.

---

## Quick start

```bash
pip install -r Sprint2/requirements.txt


python Sprint2/src/train_classifier.py --dimension type
python Sprint2/src/train_classifier.py --dimension complexity
python Sprint2/src/train_classifier.py --dimension domain

python Sprint2/src/train_classifier.py --dimension type --save LogisticRegression
```

Results land in `Sprint2/results/<dimension>_results.json`, models in
`Sprint2/models/<dimension>_classifier_baseline.pkl`.

---

## The one thing to understand before training

In the Sprint 1 dataset, **the benchmark source predicts the label almost
perfectly**:

| Predicting label from source name alone | Accuracy |
|---|---|
| Type | 99.8% |
| Domain | 95.2% |
| Complexity | 90.9% |
| Recovering the source itself from the embedding | 82.4% (majority 18.5%) |

Because the Sprint 1 split is random, rows from the same source sit on both
sides of the train/test line. A model can therefore score ~0.97 by recognising
*dataset style* rather than by learning the taxonomy.

This is not a labeling error and it does not require rebuilding Sprint 1. It is
an **evaluation** problem, and the fix is to report a second protocol.

### Two protocols, both reported

| Protocol | What it does | What it answers |
|---|---|---|
| **RANDOM** | The Sprint 1 stratified split | "How well does this do on tasks that look like our training data?" |
| **GROUPED** | Leave-one-source-out over provenance | "How well does this do on a task from a source it has never seen?" |

**GROUPED is the selection metric.** The gap between the two is a reportable
result, not a failure — a 0.97 a reviewer can break with one question is a
liability; a documented generalization gap is a finding.

### Why it matters for model selection

On the RANDOM protocol the three candidates sit within noise of each other and
picking between them is a coin flip. On GROUPED they separate decisively:

| Dimension | Model | RANDOM | **GROUPED** | Nested | Gap |
|---|---|---|---|---|---|
| Type | **LogisticRegression** | 0.961 | **0.762** | **0.715** | 0.200 |
| Type | RandomForest | 0.951 | 0.493 | — | 0.459 |
| Type | SVM | 0.986 | 0.617 | — | 0.369 |
| Complexity | **LogisticRegression** | 0.894 | **0.702** | **0.679** | 0.192 |
| Complexity | RandomForest | 0.797 | 0.312 | — | 0.485 |
| Complexity | SVM | 0.904 | 0.640 | — | 0.264 |
| Domain | **LogisticRegression** | 0.969 | **0.491** | **0.477** | 0.478 |
| Domain | RandomForest | 0.886 | 0.186 | — | 0.701 |
| Domain | SVM | 0.992 | 0.369 | — | 0.623 |

*Nested* = nested leave-one-source-out CV, where C is chosen inside each fold so
the held-out source never influences it. Quote these in the thesis. Final models
are `LogisticRegression(C=1.0)`; see `docs/phase3_model_selection.md`.

RandomForest looks competitive on RANDOM and collapses to near the majority
baseline on GROUPED. LogisticRegression generalizes best on all three
dimensions — and is also the fastest and the best calibrated.

---

## Classifier interface (Phase 0 contract)

**Input** — the shared Sprint 1 embedding. Do not re-embed.

```json
{ "task_id": "T0001", "embedding": [0.021, -0.114, "..."] }
```

`all-MiniLM-L6-v2` · 384-dim · L2-normalized · one embedding per task, shared by
all three classifiers and by the Classification Cache.

**Output** — label, confidence, **and margin**.

```json
{ "prediction": "Coding & Debugging", "confidence": 0.87, "margin": 0.34 }
```

**Saving** — `joblib`, as `{type|complexity|domain}_classifier_baseline.pkl`.
Every model exposes `.predict()` and `.predict_proba()`.

### Why `margin` was added to the contract

Decision 2 commits BudgetMind to **Margin Sampling** for cache admission: accept
a cached classification only when the top two classes are well separated. Max
confidence cannot express this. Two predictions both reporting 0.76:

```
p_top1 = 0.76, p_top2 = 0.68  ->  margin 0.08, genuinely uncertain
p_top1 = 0.76, p_top2 = 0.05  ->  margin 0.71, confident
```

Identical under a confidence-only schema. Without `margin`, Member 4 cannot
implement the committed cache design.

Measured on the test set, gating on margin does what Decision 2 predicts:

| Gate | Admitted | All three labels correct among admitted |
|---|---|---|
| ungated | 100% | 0.843 |
| min margin ≥ 0.3 | 71.1% | 0.932 |
| min margin ≥ 0.5 | 56.6% | **0.979** |

The gate depends on the *ranking* of margins, not on absolute calibration, which
is why it works well even though C=1.0 is less calibrated than C=10 (see
`docs/phase3_model_selection.md` §3).

### Label resolution

Resolve the label from `predict_proba().argmax()`, **not** from `.predict()`.
`SVC(probability=True)` derives probabilities through an internal Platt-scaling
cross-validation, so the two can disagree on rare cases. Taking the label and
the margin from the same distribution guarantees the cache and the routing
decision never contradict. `predict_with_margin()` does this for you.

---

## Task Profile

```json
{
  "task_id": "T0012",
  "taxonomy_version": "1.0",

  "type": "Factual Q&A",
  "complexity": "Low",
  "domain": "General Knowledge/Cross-Domain",

  "type_confidence": 0.9729,
  "complexity_confidence": 0.8951,
  "domain_confidence": 0.9301,

  "type_margin": 0.9632,
  "complexity_margin": 0.7915,
  "domain_margin": 0.8752,

  "embedding_ref": "emb:T0012",
  "classified_at": "2026-09-10T06:41:19Z",
  "from_cache": false
}
```

| Field | Consumer | Use |
|---|---|---|
| `type` / `complexity` / `domain` | Contextual Bandit | The context bucket (`context_bucket()`) |
| `*_margin` | Classification Cache | Margin Sampling admission (`min_margin()`) |
| `*_confidence` | Bandit | Exploration signal — low confidence means a less trustworthy prior |
| `complexity` | Budget Optimizer | Drives which tier is eligible |
| `type` / `domain` | Outcome Evaluator | Selects the scoring method |
| `taxonomy_version` | Cache | Invalidates entries when the taxonomy changes (Guidelines §2.3 anticipates extending DOMAIN) |
| `embedding_ref` | Cache | Proves classifier and cache share one vector |
| `from_cache` | Evaluation | Enables the two-metric split |

```python
from task_profile import build_task_profile

profile = build_task_profile(
    task_id, embedding,
    type_model=type_clf, complexity_model=complexity_clf, domain_model=domain_clf,
)
profile.context_bucket() 
profile.min_margin()      
```

---

## Reported metrics

Every member reports the same block, via `evaluation.format_evaluation()`:

- Accuracy, macro Precision / Recall / F1
- **Per-class** precision / recall / F1 **with support**
- **Majority-class baseline** — without it, "0.894 on Complexity" is unreadable; against 0.238 it is a result
- **ECE (calibration)** — Decision 2 thresholds these probabilities, and a threshold on uncalibrated scores is arbitrary
- Confusion matrix + every misclassified row with confidence and margin
- Both protocols, and the gap between them

### Sample-size caveat to state in the thesis

The test split is 83 rows and the smallest Type classes have 5–8 examples there.
One flipped Summarization prediction moves that class's F1 by roughly 0.1. Always
report per-class **support** beside per-class F1.

---

## Files

```
Sprint2/
├── README.md                  this document — the Phase 0 contract
├── requirements.txt
├── src/
│   ├── data.py                loads Sprint 1 embeddings; validates 384-dim + L2 norm
│   ├── evaluation.py          both protocols, majority baseline, ECE, error analysis
│   ├── task_profile.py        Task Profile + predict_with_margin()
│   └── train_classifier.py    CLI: train, compare, save
├── models/                    finalized baselines (joblib)
└── results/                   per-dimension JSON reports
```

