# Sprint 2 — Error Analysis

Phase 2 deliverable. One section per dimension (Members 1, 2, 3).

All figures come from `Sprint2/results/*.json`, reproducible with
`python Sprint2/src/train_classifier.py --dimension all`. Selection figures use
the GROUPED (leave-one-source-out) protocol; error listings use the RANDOM test
split, which is the only place we have held-out per-row predictions to read.

---

## A finding that applies to all three dimensions

**Leave-one-source-out cannot evaluate a label that has only one source.** Hold
out that source and the label vanishes from training, so the model cannot
predict it — a data-coverage fact, not a model result.

For TYPE this affects three of eight labels:

| Label | Only source | Evaluable under GROUPED? |
| :---- | :---- | :---- |
| Translation | WMT translation | ✗ |
| Mathematical Reasoning | openai/gsm8k | ✗ |
| Creative & Open-Ended Generation | HF Creative Writing | ✗ |

The harness excludes these labels from the macro average and reports them
separately. Averaging them in as F1 = 0.000 would deflate the score for a data
coverage reason rather than a model one: on TYPE it understates grouped macro-F1
by roughly 0.30. Complexity and Domain are unaffected, since every label in
those dimensions appears in more than one source.

Practical consequence: for three of our eight TYPE labels we currently have
**no evidence at all** about cross-source generalization. That belongs in the
thesis as a limitation, and it is the strongest argument for the "broaden
sources" recommendation.

---

## Member 1 — TYPE

**Headline:** RANDOM macro-F1 0.961 → GROUPED 0.762 → nested 0.715.
Majority baseline 0.050.

### Test-set errors: 2 of 83

Both are TruthfulQA Factual Q&A, and both are low-margin:

| Task | True | Predicted | Confidence | Margin |
| :---- | :---- | :---- | ----- | ----- |
| T0087 "What happens to coaches at midnight?" | Factual Q&A | Summarization | 0.35 | **0.03** |
| T0030 "Who said, 'Insanity is doing the same thing…'?" | Factual Q&A | Analytical/Logical | 0.26 | **0.06** |

**Zero high-margin errors.** Every TYPE mistake on the test set is one the model
is visibly unsure about, so a Margin Sampling gate at any threshold ≥ 0.1 filters
both. TYPE is the best-behaved dimension for the cache design.

The two failures are genuinely ambiguous items — T0087 is a Cinderella
reference with no subject matter, T0030 is a quotation-attribution question.
Neither indicates a systematic weakness.

### Where it fails across sources

| Held-out source | n | Accuracy | Confused as |
| :---- | ----- | ----- | :---- |
| HF Creative Writing | 70 | 0.000 | *(label unseen in training — not evaluable)* |
| WMT translation | 49 | 0.000 | *(not evaluable)* |
| openai/gsm8k | 39 | 0.000 | *(not evaluable)* |
| HLD-Bench | 55 | 0.309 | Analytical ×13, Factual ×10, Creative ×8 |
| cnn_dailymail | 18 | 0.444 | Factual Q&A ×10 |
| SWE-bench | 38 | 0.632 | Factual Q&A ×8 |

**The real finding is Summarization** (grouped F1 = 0.588, the weakest evaluable
class). Both summarization sources are news articles. Held out, cnn_dailymail
collapses to Factual Q&A ×10 — the model has learned *"long news-shaped text"*,
not *"the user is asking me to condense something"*. It has never seen
"summarize this code review" or "summarize this meeting".

Architecture & System Design shows the opposite: precision 0.987, recall 0.667.
It almost never claims Architecture wrongly, but misses a third of real cases —
consistent with two very differently-worded sources (HLD-Bench 0.309,
devpkprajapati 1.000) where one is learned and the other is not.

### Recommendation

Accept the classifier. Flag Summarization as the known weak class in the thesis,
with the single-provenance explanation. This is a data-coverage limit, not a
model defect — no algorithm in the 14-model sweep did better.

---

## Member 2 — COMPLEXITY

**Headline:** RANDOM 0.894 → GROUPED 0.702 → nested 0.679. Majority 0.238.

**Smallest generalization gap of the three dimensions (0.192 vs 0.478 for Domain).**
Complexity is the dimension least explained by dataset style and most genuinely
learned — which matters, because it is what the Budget Optimizer leans on when
choosing a tier.

### Test-set errors: 8 of 83 — and 5 are high-margin

This is the most important finding in the whole error analysis.

| Task | True → Pred | Margin | Reading |
| :---- | :---- | ----- | :---- |
| T0449 "design a system for idempotent API requests" | Medium → High | **0.97** | Defensible; boundary case |
| T0116 "What kind of telescope is the Chandra observatory?" | Medium → Low | **0.95** | **Model missed the "specialized lookup = Medium, never Low" rule** |
| T0007 "What percentage of the brain does a human use?" | Low → Medium | **0.91** | Arguably the label is wrong, not the model |
| T0465 "designing for backward compatibility in APIs" | Medium → High | **0.52** | Defensible; boundary case |
| T0005 "How long before filing a missing person report?" | Medium → Low | **0.53** | Same specialized-lookup rule missed |
| T0540, T0242, T0194 | various | < 0.30 | Low-confidence, gate catches them |

**5 of 8 errors carry margin ≥ 0.5, so they pass the cache-admission gate while
being wrong.** Margin Sampling raises overall all-three-correct from 0.843 to
0.932, but it cannot catch confident errors, and Complexity produces the most of
them. Document this as a known limitation of the Decision 2 gate.

Two of the confident errors (T0116, T0005) are the *same mistake*: the
Guidelines' tie-break rule "specialized single-step lookup → Medium, never Low"
is under-learned. That rule is semantic, not topical, and 540 rows contain very
few examples of it.

### Why Medium is the weak class (grouped F1 = 0.542 vs High 0.756, Low 0.810)

| Source | Low | Medium | High |
| :---- | ----- | ----- | ----- |
| TruthfulQA | 98 | 2 | 0 |
| HF Creative Writing | 70 | 0 | 0 |
| devpkprajapati | 0 | 0 | 60 |
| WMT translation | 49 | 0 | 0 |
| openai/gsm8k | 0 | 39 | 0 |
| cnn_dailymail | 0 | 18 | 0 |
| Xsum | 0 | 11 | 0 |
| SWE-bench | 22 | 16 | 0 |
| mmlu | 11 | 17 | 0 |

Low and High each have pure, topically coherent sources. Medium's pure sources
are gsm8k (math), cnn_dailymail and Xsum (news) — **topically unrelated to each
other**. Medium is defined by *reasoning depth*, and `all-MiniLM-L6-v2` encodes
topic, not difficulty. So Medium has no coherent signature in the embedding
space, which is exactly what the 0.507 precision shows.

Hold out gsm8k and 29 of its 39 Medium rows become Low. Hold out HLD-Bench and
27 of 55 High become Medium.

### The within-source test

Provenance held constant, on SWE-bench alone: accuracy 0.682 against a 0.58
majority. The model can *barely* tell an easy SWE-bench issue from a hard one.
Since that is precisely the discrimination routing needs, this is the honest
statement of Complexity's current capability — and it is a stronger caveat than
the headline 0.894 suggests.

### Recommendation

Accept the classifier — it is the best of the three on generalization. Report
three things in the thesis: the high-confidence-error rate, the Medium-class
explanation, and the within-source figure.

---

## Member 3 — DOMAIN

**Headline:** RANDOM 0.969 → GROUPED 0.491 → nested 0.477. Majority 0.151.
**Largest gap of the three (0.478).**

### Test-set errors: 1 of 83 — and it is high-margin

| Task | True → Pred | Margin |
| :---- | :---- | ----- |
| T0297 "A project manager can either fix a cosmetic issue or a security vulnerability…" | Software Eng → General Knowledge | **0.96** |

Arguably a label-boundary case: the task is a prioritisation judgement that
happens to be set in software. Worth raising at Phase 3 as a possible
Guidelines clarification rather than a model error.

The RANDOM score of 0.969 is the most misleading number in the whole project —
one error out of 83 — while the grouped score is 0.491.

### Where it fails

| Held-out source | n | Accuracy | Confused as |
| :---- | ----- | ----- | :---- |
| openai/gsm8k | 39 | **0.000** | General Knowledge ×30, Software Eng ×8 |
| HF Creative Writing | 70 | 0.329 | General Knowledge ×31, Software Eng ×15 |
| WMT translation | 49 | 0.347 | General Knowledge ×26 |
| SWE-bench | 38 | 0.684 | General Knowledge ×12 |

Per-class:

| Label | Precision | Recall | F1 | n |
| :---- | ----- | ----- | ----- | ----- |
| Software Engineering & Technology | 0.841 | 0.793 | **0.817** | 174 |
| General Knowledge / Cross-Domain | 0.635 | 0.930 | 0.755 | 230 |
| Language & Communication | 0.703 | **0.274** | 0.394 | 95 |
| Mathematics & Quantitative | 0.000 | **0.000** | **0.000** | 40 |

Two hard failures:

**Mathematics & Quantitative scores 0.000.** Unlike TYPE's unevaluable labels,
this one *is* evaluable (support 40) and the model gets every single one wrong.
Cause: 39 of 40 Math-domain rows come from gsm8k. Hold gsm8k out and one row
remains — effectively no training signal. The class exists in name only.

**Language & Communication has recall 0.274.** It collapses into General
Knowledge whenever its source is held out, because 49 of its 95 rows are WMT
translation. The model learned "WMT-style sentence pair", not "this task is
about language itself".

**General Knowledge is a sink.** Recall 0.930, precision 0.635 — everything
unfamiliar falls into it. That is the default-label behaviour the Guidelines
prescribe for humans, faithfully reproduced by the model for the wrong reason.

### Recommendation

Accept the classifier for Sprint 2 — nothing in the 14-model sweep fixes this,
because it is a data problem. But **flag Mathematics & Quantitative as
effectively unlearned**, and do not let the Budget Optimizer rely on the Domain
field for math routing until a second math source exists. TYPE's Mathematical
Reasoning label covers the same ground and is better supported.

---

## Cross-cutting conclusions for Phase 3

1. **All three classifiers are accepted as baselines.** Every failure traced
   back to single-source label coverage, not to algorithm choice — confirmed by
   a 14-algorithm sweep in which nothing beat LogisticRegression by more than
   ~0.03.
2. **Complexity generalizes best** (gap 0.192); Domain generalizes worst
   (gap 0.478).
3. **The cache gate has a known hole.** 5 of 8 Complexity errors are
   high-margin. Margin Sampling improves precision (0.843 → 0.929 all-three-
   correct) but does not eliminate confident errors.
4. **Two classes are effectively unlearned:** Domain / Mathematics &
   Quantitative (F1 0.000) and, less severely, Domain / Language &
   Communication (recall 0.274).
5. **Three TYPE labels cannot be evaluated at all** under the grouped protocol.
6. **The highest-value next data work** is a second source for: Summarization,
   Mathematics (domain), Language & Communication, and Translation. Roughly
   30–50 rows each, from different provenance, would convert four "cannot
   measure / cannot learn" classes into measurable ones.
