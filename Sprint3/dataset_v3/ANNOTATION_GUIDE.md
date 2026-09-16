# Complexity annotation — independent verification pass

**You must not have been involved in building v2.1.** If you wrote, selected, or
scored any of these rows, you are not eligible to be an annotator on this pass.

**Status: NOT STARTED. No human annotations exist yet. No κ has been computed.**

---

## 1. What you see

A blind sample would present two columns:

| Column | Meaning |
|---|---|
| `row_ref` | opaque reference (R001…R180). Carries no information. |
| `task_text` | the task exactly as a user would send it |

Everything else is blank and is for you to fill in.

## 2. What an annotator must NOT see

Withheld because each would bias the judgement:

- the **source** the row came from (this is the single most important omission — source identity predicts the existing label 78% of the time, so seeing it would let you reproduce the label without judging the task)
- the **existing complexity band**
- the **existing rubric scores**
- the TYPE and DOMAIN labels
- whether the row is inherited from v1 or new in v2
- any per-row notes from construction

`sample_key_DO_NOT_OPEN_BEFORE_ANNOTATING.csv` contains the mapping. Opening it
before you finish invalidates the whole pass. It exists so the κ script can join
your work to the existing labels afterwards.

## 3. How to score the six dimensions

Read the task. Ask only: *what must a competent solver actually do?* Ignore who
wrote it, where it came from, how long it is, and how technical it sounds.

Score each 0–2:

| Dim | Question | 0 | 1 | 2 |
|---|---|---|---|---|
| **D1** steps | How many distinct reasoning operations? | one | 2–3 | 4+ |
| **D2** dependency | Do later steps consume earlier results? | independent/single | one chained handoff | chain of 2+, or an early error invalidates everything after |
| **D3** synthesis | How far is the output form from the input form? | restate/retrieve | reorganise, condense or convert within one frame | combine multiple sources or frames into something neither contains |
| **D4** ambiguity | How much must the solver decide before starting? | fully specified | one choice to make and justify | goal, scope or success criteria are themselves unstated |
| **D5** constraints | Simultaneous requirements that can conflict? | none | 1–2 compatible | 3+, or any two that trade off |
| **D6** error cost | What does a wrong answer cost, and is it catchable? | obvious if wrong, trivial to fix | needs checking, moderate rework | plausible-but-wrong is likely **and** either the cost is real (safety/legal/financial/data-loss/production) **or** verifying costs about as much as producing |

Then score **K** separately (it does **not** add to the total):

- `K=0` common or encyclopedic knowledge
- `K=1` practitioner knowledge — only if **all three** hold: a well-read layperson could not answer reliably without professional training or reference; acting on a wrong answer carries real cost; a non-expert could not tell a correct answer from a confidently wrong one
- `K=2` the answer requires professional *judgment*, not just professional *facts* — two qualified practitioners could reasonably disagree

## 4. How to assign the band

Sum D1–D6, then read off:

| Total | Band |
|---|---|
| 0–3 | Low |
| 4–7 | Medium |
| 8–12 | High |

Then apply, in order, and note which fired:

1. any `D6=2` → at least Medium
2. `D4=2` → at least Medium
3. `D2=2` **and** `D5=2` → at least High
4. `K=1` → at least Medium; `K=2` → at least High

Floors only ever raise a band. Length never adjusts a score. Subject prestige
never adjusts a score.

## 5. Ambiguity

Set `ambiguous_yes_no = yes` and write one line in `annotator_note` when:

- the task is ill-posed or internally contradictory
- a key term has no stated referent ("which is larger", "which is better")
- the task references material that was not supplied
- you genuinely cannot choose between two adjacent bands

**Still record your best scores.** Do not leave a row blank and do not split the
difference — pick the band you would defend, and flag it.


---

## 6. Outstanding work

An independent two-annotator pass over a blind, stratified sample remains **outstanding**.
It is the one gate that would move the corpus past
`READY_WITH_DOCUMENTED_LIMITATIONS`. The sample and the κ tooling were prepared and then
removed in the final cleanup as process artefacts; both are regenerable from
`complexity_dataset_v3_3.jsonl`.

Thresholds to apply when it is run (project Phase 2 protocol, unchanged):
calibration κ ≥ 0.70, per-batch κ ≥ 0.75. Report Cohen's κ for the band (nominal) and
linearly weighted κ for each of D1–D6 and K (ordinal).
