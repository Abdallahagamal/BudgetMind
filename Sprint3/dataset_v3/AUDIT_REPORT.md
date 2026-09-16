# Complexity Dataset v3 — full audit report

> **Note (final cleanup):** this audit was run on `complexity_dataset_v3.jsonl` and produced
> `complexity_dataset_v3_3.jsonl`; both were removed once v3.2 became canonical. The corrections it
> describes are all present in `../dataset_v3/complexity_dataset_v3_3.jsonl`, and the per-row
> decisions are in `human_adjudication_log.jsonl`.

**Canonical input:** `Sprint3/dataset_v3/complexity_dataset_v3.jsonl` (1008 rows) — the only
v3 corpus file present; confirmed against `Sprint3/dataset_v3/README.md`.
**Output:** `complexity_dataset_v3_3.jsonl` (1008 rows). The original v3 file is
unmodified.

## Status of the A/B files

`annotator_A.csv` and `annotator_B.csv` are **simulated model passes**, not human
annotators. They are used here only as *adjudication pass A* and *adjudication pass B* —
a device for surfacing rows where two readings of the rubric diverge. No κ from them is
treated as reliability evidence, and none is used to justify a label change.

## 1. Inventory and reconciliation

| | |
|---|---|
| rows / unique task_id | 1008 / 1008 |
| sources | 22 |
| Low / Medium / High | 470 / 382 / 156 |
| v1-inherited / v2–v3 authored | 503 / 505 |
| rows with stored rubric scores | 505 |

Changelog reconciles exactly: 973 − 37 removals + 72 additions = 1008. Removals are all
devpkprajapati concept-duplicates; additions are MATH 30, OASST1 27, StackExchange 15.
Zero relabels in v3, as claimed.

## 2. Integrity findings

| Check | Result |
|---|---|
| Empty / control chars / broken encoding / metadata-in-text / self-duplicated text | **0** |
| Hidden Unicode | **0** (the one known case is in `blocklist.json` and stayed out) |
| Exact duplicate texts | **0** |
| Rubric band vs stored D1–D6/K | **0 mismatches** in 505 scored rows |
| Stored total vs recomputed total | **0 mismatches** |
| Scores out of 0–2 range | **0** |
| Truncation markers | 20 (18 BillSum, 2 StackExchange) — documents clipped at the build cap |

## 3. Duplicates — zero removals justified

31 within-source near-duplicate pairs at Jaccard ≥ 0.75. **All are legitimate:**

- **DROP (20 pairs)** — same passage, different question, different `query_id`. Verified
  pairwise: the questions genuinely differ. Five pairs carry different labels, correctly.
- **WMT (9 pairs)** — same English source sentence, different target language. Different tasks.
- **TruthfulQA (1)** — "what happens to your pants / nose if you tell a lie". Different questions.
- **SVAMP (1)** — different quantities and a different question.

No row was removed for redundancy.

## 4. Source leakage

30 n-grams identified a source at ≥0.90 precision *and* ≥0.90 recall. Split:

- **Accidental construction artefact (12, fixed):** `cnn_dailymail` carried the constant
  prefix `"Summarize this article: "` on all 18 rows plus scrape boilerplate
  (`E-mail to a friend .`, `Watch …»`); `Xsum` carried `"Summarize this document. "` on all
  11. This is the same defect class fixed for BillSum in v2.1 but never applied to
  v1-inherited rows. **Text cleaned on 29 rows; no label changed.** Leakage n-grams 30 → 18.
- **Legitimate task content (18, kept):** WMT `"translate the following"` *is* the task;
  BillSum `"section 1"` is bill structure; devpkprajapati's `p99` / `targeting` template is
  task content and is separately documented.

## 5. The BillSum rubric question

The rubric does **not** define how to treat a task that references material not supplied.
An amendatory bill says "Section 219 … is amended by striking paragraph (5)" — summarising
what it *does* depends on text that is not in the row.

Proposed clarification (**not applied to the rubric here — for team sign-off**):

> **Unsupplied referenced material.** If a task references material that is not included:
> (a) if that material is *essential* and the task cannot be completed without it, score
> **D4 = 2** and flag ambiguous; (b) if the task remains completable because the relevant
> content is quoted inline, but verifying the answer requires the external source, score
> **D6 = 2** and leave D4 unchanged.

Amendatory bills fall under (b) — the inserted text is quoted, so the task is completable,
but verification needs the amended statute. This supports the existing High labels on
densely amendatory bills and Medium on lighter ones. **No BillSum label was changed on this
basis.**

## 6. What was changed

**9 label corrections, 14 score-only corrections, 29 text cleanups, 0 removals.**

### 6a. K-test failures (2) — Medium → Low
`T0116` "What kind of telescope is the Chandra observatory?" and `T0119` "The point of
concurrence of medians is called?" fail **all three** parts of the K specialisation test in
`ANNOTATION_GUIDE.md` §3 (practitioner / consequence / verification). Medium is only
reachable for these via K=1. Both are minority Mediums among 31 Low NQ rows, while
comparable NQ items (e.g. `T0104`, a legislative date) are Low. The Phase 2 specialisation
review had already proposed exactly these two.

### 6b. Internal contradictions (2) — Medium → Low
`V0083`, `V0098` carry **D3 = 2** while 22 structurally identical short P3 comparison rows
carry D3 = 1. Two retrievals followed by a comparison is synthesis within one frame, not
across frames. Recomputed total 3 → Low.

### 6c. BillSum score artefact (5 band moves + 14 score-only) — Low → Medium
The v2.1 `bill_scores` rule set `D2` from the literal string `"is amended"`, so
self-contained bills scored **D2 = 0** regardless of content. Every BillSum row is ≥2822
characters with multiple provisions; "independent or single step" is unsupported for any of
them. D2 0→1 on 19 rows and D3 1→2 on 3 rows; the band moves only for the 5 rows that were
sitting in Low (`V0146`, `V0147`, `V0151`, `V0158`, `V0165`).

### Direction check
4 changes move Medium→Low and 5 move Low→Medium. Net class shift: Low 470→469,
Medium 382→383, High unchanged. **No change was made in a direction that improves balance,
accuracy or the acceptance margin**, and nothing was removed.

## 7. What was deliberately NOT changed

- **Three disputed rows where both readings are defensible** — `V0407` (perovskite intro),
  `V0435` (3D-print warping), `V0275` (mysqldump optimisation). Pass A read D2=2, pass B
  read D2=1; the totals straddle the band boundary. Changing them would substitute one
  model judgment for another, not correct an error. Marked `KEEP_LOW_CONFIDENCE`.
- **9 devpkprajapati rows with slack SLOs** (p99 1500–6000 ms) labelled High alongside rows
  at p99 < 40 ms. The task variation may not support a single band, but these are
  v1 human labels with no stored scores to contradict. Marked `NEEDS_HUMAN_REVIEW`.
- **`T0500`, `T0479`** (creative, v1 Low; rubric re-derivation gives Medium via D5=1 for the
  3-sentence / limerick constraint) — model judgment against a human label. Flagged, not changed.
- **`V0193`** — ill-posed SVAMP item, retained as a legitimate robustness example.
- **WMT / GSM8K / SVAMP uniform distributions** — preserved. Their purity is a genuine
  property of the tasks, not a construction artefact.

## 8. Decision counts

| Decision | Rows |
|---|--:|
| KEEP_CONFIRMED | 946 |
| NEEDS_HUMAN_REVIEW | 30 |
| KEEP_LOW_CONFIDENCE | 23 |
| CHANGE_LABEL | 9 |
| REMOVE_INVALID | 0 |
| DUPLICATE_OR_REDUNDANT | 0 |

`KEEP_CONFIRMED` rests on: zero integrity flags, band recomputing exactly from stored
D1–D6/K where scores exist, and peer-consistency within source and pattern where they do
not. It is not a claim that 946 rows were each individually re-argued from scratch.

## 9. Readiness

See `TRAINING_READINESS.md`.
