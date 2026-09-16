# Dataset v3 — training readiness

**Canonical build: v3.3** (1008 rows).

**Canonical corpus:** `Sprint3/dataset_v3/complexity_dataset_v3_3.jsonl` (1008 rows)
**Training-ready embeddings:** `Sprint3/dataset_v3/embeddings_v3_3/` (train 706 / validation 153 / test 149) — gitignored, regenerate with `src/make_embeddings.py`
**Runner:** `python Sprint3/dataset_v3/src/train_on_v3.py --dimension all --save LogisticRegression`

Sprint 1 and Sprint 2 are unmodified. The runner imports `Sprint2/src` and only redirects
the embeddings directory, so the committed pipeline is reused as-is.

## How the labels were produced

Labels were assigned by **automated rubric scoring**: each task is scored on six reasoning
dimensions (D1–D6) plus a knowledge flag (K) defined in `ANNOTATION_GUIDE.md`, and the
Low/Medium/High band is derived from those scores by fixed rules rather than assigned
directly. Every scored row stores its own `rubric_scores`, so any label can be checked
against the rubric that produced it — `src/audit_mechanical.py` verifies that all 511
scored rows recompute to their stated band.

Rows inherited from Sprint 1 (503) carry the original human labels and are marked
`inherited_unmodified`. Rows where the rubric was ambiguous were routed to human
adjudication; 47 are marked `human_adjudicated`, with the decision and rationale for each
recorded in `human_adjudication_log.jsonl`. The remaining rows are marked
`provisional_pending_human_verification`.

This is a model-assisted annotation pipeline with a documented rubric and an auditable
score trail. Its main open item is stated in limitation 4 below.

## Integrity gates — all pass

| Check | Result |
|---|---|
| Rows / unique `task_id` | 1008 / 1008 |
| Stored rubric scores recompute to totals | 511 scored rows, 0 mismatches |
| Totals map to bands | 0 mismatches |
| Exact duplicate texts | 0 |
| Hidden Unicode / empty text / truncation markers | 0 |
| Near-duplicate clusters spanning splits | 0 (31 pairs, all within-split) |
| Source-identifying n-grams | 18, all legitimate task content |
| Embeddings 384-dim, L2-normalised | verified, ‖x‖ = 1.000000 |
| `Sprint2/src/data.py` loader validation | passes for all three splits |

## Final evaluation — all three dimensions

MiniLM + `LogisticRegression(class_weight='balanced', C=1.0, seed 42)`.

| Dimension | RANDOM macro-F1 | GROUPED macro-F1 | Gap | Majority |
|---|--:|--:|--:|--:|
| TYPE | 0.878 | 0.543 | 0.335 | 0.043 |
| COMPLEXITY | 0.663 | 0.530 | 0.133 | 0.209 |
| DOMAIN | 0.917 | 0.725 | 0.192 | 0.144 |

**Every label in every dimension is evaluable under grouped leave-one-source-out.** In
Sprint 2, three TYPE labels (Translation, Mathematical Reasoning, Creative) could not be
scored at all because each existed in only one source, and Domain/Mathematics scored
F1 = 0.000. Both defects are gone.

Per-class grouped F1 worth noting: Domain/Mathematics & Quantitative **0.766** (was 0.000);
Domain overall **0.725** (was 0.491); Complexity/High **0.577**, Medium **0.452**.

## Known limitations — read before training

1. **TYPE and DOMAIN labels on 505 rows were never audited.** The audit covered COMPLEXITY
   only. On rows added in v2/v3 the type and domain values were assigned as a by-product of
   complexity annotation and have had no review pass. Treat TYPE and DOMAIN results on this
   corpus as provisional in a way the complexity results are not.
2. **TYPE / Translation is effectively single-source** — 49 of 50 rows are WMT. Under LOSO it
   scores F1 **0.037** (recall 0.020): hold out WMT and one training example remains. This
   is a coverage defect, not a model defect. TYPE / Summarization is also weak (0.242).
3. **The complexity acceptance gate still fails**: model 0.530 vs TYPE-lookup rule 0.581,
   margin **−0.052**. Unchanged criterion, unchanged verdict.
4. **Label validation is partial.** 47 rows were adjudicated by a single reviewer; 475
   remain `provisional_pending_human_verification`. Because there was one reviewer rather
   than two independent ones, no inter-annotator agreement was measured and no κ is
   claimed. Closing this means annotating a stratified sample under the procedure in
   `ANNOTATION_GUIDE.md` §6 and reporting κ against the stated thresholds.
5. **RANDOM complexity fell to 0.663** (v1 reported 0.894). This is expected — the earlier
   figure was inflated by source–label confounding that has since been reduced.

## Verdict

**READY_FOR_TRAINING on COMPLEXITY, with documented limitations.**
**READY_FOR_TRAINING on DOMAIN, with the caveat in (1).**
**TYPE is trainable but Translation will not generalise** until a second translation source
is added; report its per-class F1 rather than the macro average alone.

Status remains **READY_WITH_DOCUMENTED_LIMITATIONS** overall — limitation (4) has not been
cleared and no independent validation process has been completed.




---

# v3.3 — opus_books removed for public redistribution

The 30 `opus_books` rows added in v3.2 were removed. That source is licensed "other":
research and educational use only, with commercial use and mass redistribution not granted.

| Dimension (grouped macro-F1) | v3.2 | v3.3 |
|---|--:|--:|
| TYPE | 0.667 | **0.543** |
| DOMAIN | 0.742 | **0.725** |
| COMPLEXITY | 0.530 | **0.530** |

| TYPE per-class grouped F1 | v3.2 | v3.3 |
|---|--:|--:|
| **Translation** | 0.899 | **0.037** |
| Creative & Open-Ended | 0.850 | 0.735 |

**This is a real regression, accepted deliberately in exchange for redistributability.**
Translation is again 49/50 rows from WMT. Restoring it needs a second translation source
under a permissive licence — FLORES-200 and Tatoeba-MT are gated, and `opus-100` has an
unknown licence, so this remains open work.

## Redistribution status

Removing `opus_books` cleared the research-only source, but the corpus is **not yet clear
for public release**:

1. **14 of 22 sources have no licence on record** (503 rows, all inherited from Sprint 1):
   TruthfulQA, HF Creative Writing, HLD-Bench, WMT, gsm8k, SWE-bench, Natural Questions,
   mmlu, devpkprajapati, MBPP, cnn_dailymail, BIG-Bench-Hard, Xsum, Original.
2. **cnn_dailymail (18) and Xsum (11) embed verbatim news article text** whose copyright
   sits with the publishers.
3. **CC BY-SA sources impose ShareAlike** — any public redistribution must be under a
   compatible licence and must carry `ATTRIBUTION.md`.

**Status: READY_WITH_DOCUMENTED_LIMITATIONS for internal use and thesis evaluation.
NOT cleared for public redistribution until items 1 and 2 are resolved.**
