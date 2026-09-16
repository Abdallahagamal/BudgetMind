# Sprint 3 — TYPE Classifier (Eraa)

This README explains what does each file do and what is its output, how to run it, and what it
means

## What this sprint did

Three things, matching the sprint plan's Member 1 tasks:

1. **Model refinement** — calibrated the Type classifier's confidence scores
   (Sprint 2's own error analysis flagged this as the Sprint 3 task for
   Type specifically, since Type had the worst-calibrated probabilities of
   the three dimensions).
2. **Test classification accuracy** — verified the classifier still performs
   exactly as Sprint 2 documented, in this environment, before we build
   anything else on top of it.
3. **Shared performance sync contribution** — a standardized results file,
   in the same format Complexity (M2) and Domain (M3) will each produce, so
   the three can be merged into one report.

Plus an extra: a handoff doc of Type-specific edge cases for M5's
end-to-end testing.

---

## Folder structure

This needs to sit inside the main project repo, alongside the existing
`Sprint1/` and `Sprint2/` folders — it reads Sprint 1's embeddings and
Sprint 2's trained model directly, it doesn't duplicate them:

```
BudgetMind/                              ← repo root — run all commands from here
├── Sprint1/
│   └── embeddings_output/               ← Sprint 1's embeddings (unchanged)
├── Sprint2/
│   ├── src/                             ← Sprint 3 imports this code directly
│   ├── models/type_classifier_baseline.pkl
│   └── results/type_results.json
└── Sprint3/
    ├── Validate type Classifier/        ← the 3 scripts (this folder can be named
    │   ├── calibrate_type.py               anything — the scripts figure out where
    │   ├── test_accuracy_type.py           they are relative to the repo root)
    │   └── performance_sync_type.py
    ├── models/                          ← created automatically when you run the scripts
    │   └── type_classifier_refined.pkl
    ├── results/                         ← created automatically
    │   ├── type_refinement_report.json
    │   ├── type_accuracy_test.json
    │   └── type_performance_sync.json
    └── docs/
        ├── sprint3_type_report.md       ← full write-up with numbers/reasoning
        └── type_edge_cases_for_e2e.md   ← handoff doc for M5
```

لو ال depth بتاع الفولدر اللى فيه فيلات البايثون اتغير الكود ممكن يبوظ
**Important:** the scripts don't need to be in a folder called `src` —
`Validate type Classifier` works fine, and so would any other name. What
matters is the *depth*: the scripts need to be exactly two folders below the
repo root (`Sprint3/<any-name>/script.py`), because that's how they locate
`Sprint2/` and `Sprint1/` relative to themselves. Don't move the scripts
directly into `Sprint3/` (one level, not two) or nest them any deeper — that
would break the auto-location.

---

## Running it — every time

From the repo root : (i've already ran the code)

```bash
python "Sprint3/Validate type Classifier/test_accuracy_type.py"
python "Sprint3/Validate type Classifier/calibrate_type.py"
python "Sprint3/Validate type Classifier/performance_sync_type.py"
```

Each script prints its results to the terminal as it runs, and also saves a
`.json` report into `Sprint3/results/`.

---

## What each file does

### `test_accuracy_type.py` — run this first
A regression check, not new modeling work. It re-evaluates Sprint 2's
already-trained model on the test set, right now, in your environment, and
confirms the numbers match what Sprint 2 recorded. This exists to catch
silent problems — e.g. a different scikit-learn version quietly changing
how the model behaves, a corrupted model file, or someone accidentally
editing the embeddings. It also does a couple of basic sanity checks
(embeddings are still 384 dimensions and normalized, the test set is still
83 rows) and confirms the model's confidence/margin outputs are valid.

**Output:** `Sprint3/results/type_accuracy_test.json` — should show
`"all_checks_passed": true`. If anything says `false`, something in the
pipeline changed since Sprint 2 and needs investigating before trusting any
further results.

### `calibrate_type.py` — the model refinement
This is the actual "improve the model" work for the sprint. Some background
first: when the classifier makes a prediction, it also reports a
*confidence* number (e.g. "73% sure it's Factual Q&A"). Ideally, when the
model says 73%, it should genuinely be right about 73% of the time on
predictions like that — that property is called **calibration**. Sprint 2's
model was accurate, but its confidence numbers weren't well calibrated
(a documented issue in Sprint 2's own findings).

This script fixes that using a standard technique (`CalibratedClassifierCV`)
that adjusts the model's confidence outputs to better reflect reality,
**without retraining the model from scratch** — it wraps the existing
model and recalibrates its probability outputs. It tries a couple of
calibration methods, picks the better one using the validation set only
(not the test set — see the "why validation vs test" section in
`docs/sprint3_type_report.md` if curious), then reports the before/after
comparison using the test set exactly once at the end.

**Result:** the refined model's confidence scores are far more trustworthy
(a 62% reduction in calibration error) at a very small accuracy cost (about
1 extra wrong prediction out of every 83). The practical payoff: more tasks
can be confidently auto-routed by the system without the accuracy dropping
— see `docs/sprint3_type_report.md` for the exact numbers.

**Output:**
- `Sprint3/models/type_classifier_refined.pkl` — the new, calibrated model
- `Sprint3/results/type_refinement_report.json` — full numeric comparison

**Note:** this does *not* replace Sprint 2's baseline model
(`Sprint2/models/type_classifier_baseline.pkl` is left untouched). Whether
to actually switch the production classifier to this refined version is a
team decision — bring it to the shared sync, don't just swap it in.

### `performance_sync_type.py` — for the team sync
Produces one clean, standardized summary of Type's performance (accuracy,
precision, recall, F1, calibration error) for both the original and refined
models. It's built to match the same format M2 and M3 will produce for
Complexity and Domain, so all three can be dropped into one shared report
without anyone reformatting anything by hand.

**Output:** `Sprint3/results/type_performance_sync.json`

### `docs/sprint3_type_report.md`
The full write-up — all the reasoning and numbers above, in one document,
meant to be read rather than run. Good starting point if you only read one
file.

### `docs/type_edge_cases_for_e2e.md`
For M5 (end-to-end testing), not really needed by anyone else. Lists
specific tricky inputs for the Type classifier — known error cases,
categories with weak or no test coverage, and boundary cases — so M5's
end-to-end tests actually exercise the classifier's real weak points instead
of only easy/obvious inputs.

---

## The headline result, if you only want the takeaway

The Type classifier's **accuracy is essentially unchanged** (96.4% either
way), but its **confidence scores got much more trustworthy** (calibration
error down 62%). Practically: at the same reliability bar the system
already uses, the refined model lets noticeably more tasks be confidently
auto-classified — 78 out of 83 test tasks clear a "high confidence" bar at
100% accuracy, up from 70 out of 83 before. That's the concrete benefit for
the rest of the pipeline (routing, caching) once the team decides whether to
adopt it.

## What's not decided yet / open questions for the team

- **Whether to switch production to the refined model.** The comparison is
  documented, the decision isn't made.
- Waiting on M2 and M3 to produce their own `performance_sync` files so all
  three can be merged.
- The Summarization category remains a known weak spot (too little training
  data, not a bug) — unaffected by this sprint's work, flagged again for
  visibility.
