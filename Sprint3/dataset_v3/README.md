# BudgetMind — Complexity/Type/Domain Dataset v3.3 (canonical)

**`complexity_dataset_v3_3.jsonl` — 1008 rows, 22 sources.** The single canonical dataset
for Sprint 3. Sprint 1 and Sprint 2 are unmodified; the training runner imports
`Sprint2/src` and only redirects the embeddings directory.

v3.3 removes the 30 `opus_books` rows that were present in v3.2. That source was licensed
"other" — research and educational use only, with mass redistribution not granted — so it
blocked public release. See `SOURCES.json` for the full record and for the licensing work
that is **still outstanding**.

Labels are produced by automated rubric scoring against `ANNOTATION_GUIDE.md`: six
reasoning dimensions plus a knowledge flag, with the Low/Medium/High band derived from
those scores by fixed rules. Every scored row carries its own `rubric_scores`, so each
label is checkable against the rule that produced it. Ambiguous rows were routed to human
adjudication (`human_adjudication_log.jsonl`). See `TRAINING_READINESS.md` for the
validation status.

## Files

| File | Contents |
|---|---|
| `complexity_dataset_v3_3.jsonl` | **canonical corpus** — 1008 rows, three labels, rubric scores, provenance, splits |
| `models_v3_3/` | trained TYPE / COMPLEXITY / DOMAIN classifiers (LogisticRegression) |
| `results_v3_3/` | per-dimension results, both protocols, all three candidate models |
| `final_evaluation_v3_3.json` | headline figures per dimension + acceptance gate |
| `embeddings_v3_3/` | generated MiniLM embeddings (gitignored — rebuild with `src/make_embeddings.py`) |
| `CHANGELOG.jsonl` | every added and removed row with reason and evidence (169 entries) |
| `human_adjudication_log.jsonl` | 67 adjudication decisions (single adjudicator) |
| `SOURCES.json` | per-source licence status **and the outstanding redistribution blockers** |
| `ATTRIBUTION.md` | CC BY-SA attribution for 89 Stack Exchange posts + licence table |
| `ANNOTATION_GUIDE.md` | the rubric — required to read any `rubric_scores` field |
| `guideline_amendments.jsonl` | A1–A8, all approved; the applied rubric differs from the original text |
| `AUDIT_REPORT.md` | the audit method, findings, and every correction made |
| `TRAINING_READINESS.md` | evaluation results, limitations, readiness verdict |
| `blocklist.json` | permanently rejected source rows |

## Run it

```bash
python Sprint3/dataset_v3/src/make_embeddings.py                                    # once, after clone
python Sprint3/dataset_v3/src/train_on_v3.py --dimension all --save LogisticRegression
python Sprint3/dataset_v3/src/final_evaluation.py
python Sprint3/dataset_v3/src/audit_mechanical.py                                   # integrity check
```

## Results (grouped leave-one-source-out)

| Dimension | test acc | RANDOM macro-F1 | GROUPED macro-F1 |
|---|--:|--:|--:|
| TYPE | 0.859 | 0.878 | 0.543 |
| COMPLEXITY | 0.685 | 0.663 | 0.530 |
| DOMAIN | 0.906 | 0.917 | 0.725 |

Complexity acceptance gate: model 0.530 vs TYPE-lookup rule 0.581 — **FAIL** (margin −0.052).

**TYPE/Translation scores F1 0.037.** 49 of its 50 rows come from WMT, so holding WMT out
leaves one training example. The second translation source that fixed this in v3.2 was
`opus_books`, which had to be removed for licensing. Fixing Translation again requires a
permissively-licensed second translation source.
