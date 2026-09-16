# TYPE classifier — edge cases for M5's end-to-end testing

From Member 1 (Type Classifier), for M5's "End-to-end functional testing" —
these are the cases most likely to expose a bug somewhere in the pipeline
(cache, worker, routing), not necessarily the classifier itself. Grouped by
why they're worth including.

## 1. The two known Sprint 2 test-set misclassifications

Use these to confirm the pipeline *behaves correctly when the classifier is
wrong* — i.e. that a wrong label doesn't crash anything downstream, and that
low-margin predictions are actually being flagged as low-margin.

| task_id | Text | True | Predicted | Confidence | Margin |
|---|---|---|---|---|---|
| T0087 | "What happens to coaches at midnight?" | Factual Q&A | Summarization | 0.35 | **0.03** |
| T0030 | "Who said, 'Insanity is doing the same thing…'?" | Factual Q&A | Analytical/Logical | 0.26 | **0.06** |

Both errors are low-margin (< 0.1) — assert that both would be caught by any
margin gate you implement (min_margin ≥ 0.1 catches both). If your gate
*doesn't* catch them, that's a bug worth flagging back to me.

## 2. Summarization — the known weak class

This is a genuine, documented weakness (not a bug): the classifier has only
seen Summarization examples that are news-article summaries, so it
misreads other kinds of summarization requests as Factual Q&A. Good stress
tests for the pipeline — expect these to plausibly get misclassified as
Factual Q&A, and confirm the system still behaves sanely (doesn't crash,
still returns a valid TaskProfile):

- "Summarize this code review thread for me."
- "Give me a summary of this meeting transcript."
- "Can you condense this email chain into three bullet points?"

## 3. Zero cross-source evidence — include at least one of these

Three TYPE labels have never been validated on data from a second source
(see `docs/error_analysis.md`). We have no evidence how they generalize
beyond their one training source. Worth at least a smoke test each:

- **Translation** (only WMT-style data): `"Translate 'good morning' into French."`
- **Mathematical Reasoning** (only gsm8k-style word problems): `"If a train travels 60 miles in 1.5 hours, what is its average speed?"`
- **Creative & Open-Ended Generation** (only one creative-writing source): `"Write a short poem about autumn."`

## 4. Margin-gate boundary cases

For exercising the cache-admission logic (Decision 2 / Margin Sampling)
specifically — pick a couple of inputs you expect to be genuinely ambiguous
between two TYPE labels, and confirm the returned `margin` is low (this
tests that margin is actually discriminating, not just always high):

- "Explain how binary search works and write a Python implementation." (Coding & Debugging vs Analytical/Logical Reasoning)
- "Compare the tradeoffs of microservices vs a monolith." (Architecture & System Design vs Analytical/Logical Reasoning)

## 5. Schema checks to run on every prediction, not just the interesting ones

Whatever inputs you use, assert on the `TaskProfile` output itself:
- `type` is one of the 8 known labels (Analytical/Logical Reasoning, Architecture & System Design, Coding & Debugging, Creative & Open-Ended Generation, Factual Q&A, Mathematical Reasoning, Summarization, Translation) — a 9th value means something upstream broke
- `type_confidence` and `type_margin` are both in `[0, 1]`
- `type_margin <= type_confidence` always (margin is confidence minus runner-up, so it can never exceed confidence)
- Label is read from `predict_proba().argmax()`, not `.predict()` — call `predict_with_margin()` from `task_profile.py` rather than the raw model, so this is handled for you

## Which model to point your tests at

Sprint 3 introduced a **refined (calibrated)** version of the Type classifier
alongside Sprint 2's original baseline — see `Sprint3/results/type_refinement_report.json`
for the comparison. The refined model isn't the production baseline yet (that
swap needs a team decision), so unless told otherwise, keep testing against
`Sprint2/models/type_classifier_baseline.pkl` — the one currently deployed.
