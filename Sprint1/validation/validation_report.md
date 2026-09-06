# Labeled dataset validation report

Source file: `C:\Users\20106\OneDrive - Cairo University - Students\Desktop\BudgetMind\Sprint1\Label\labeled_dataset.csv`
Rows: **540** (plus header). Member 4 should treat this as the full labeled set to check, not 541 separate manual reads.

## What this script can and cannot do

| Automatable | Still needs a human |
|---|---|
| Missing labels | Subtle wrong TYPE/COMPLEXITY on an otherwise well-formed row |
| Values outside the taxonomy | Tie-break cases (code review vs architecture) |
| Exact duplicates | Whether a near-duplicate *should* be kept (WMT / architecture variants) |
| Near-duplicates with different labels | Gold-example style judgment |
| Source vs typical TYPE | Final accept/reject of a warning |
| Class imbalance counts | Whether to collect more data |

## Status

- Errors: **0** (must fix before splitting)
- Warnings: **10** (send to Member 1 / 3)
- Info: **76**
- Near-duplicate pairs (Jaccard ≥ threshold): **76**

See `validation_issues.csv` for every flagged `task_id`.

## TYPE

| Label | Count |
|---|---:|
| Factual Q&A | 135 |
| Architecture & System Design | 114 |
| Creative & Open-Ended Generation | 70 |
| Coding & Debugging | 58 |
| Translation | 49 |
| Analytical/Logical Reasoning | 46 |
| Mathematical Reasoning | 39 |
| Summarization | 29 |

Smallest TYPE class: **29**. If this is below ~15, stratified train/val/test is required and that class will be weak.

## COMPLEXITY

| Label | Count |
|---|---:|
| Low | 304 |
| Medium | 127 |
| High | 109 |

## DOMAIN

| Label | Count |
|---|---:|
| General Knowledge/Cross-Domain | 231 |
| Software Engineering & Technology | 174 |
| Language & Communication | 95 |
| Mathematics & Quantitative | 40 |

## SOURCE

| Label | Count |
|---|---:|
| TruthfulQA | 100 |
| Hugging Face Creative Writing datasets | 70 |
| devpkprajapati/ai-system-design-instruct | 60 |
| HLD-Bench | 55 |
| WMT translation | 49 |
| openai/gsm8k | 39 |
| SWE-bench | 38 |
| Natural Questions (NQ) | 35 |
| mmlu | 28 |
| google-research-datasets (MBPP) | 20 |
| cnn_dailymail | 18 |
| BIG-Bench-Hard | 16 |
| Xsum | 11 |
| Original | 1 |

## Manual review sample

`manual_review_sample.csv` is a **seeded stratified sample** of 64 rows (by TYPE).
Open it, apply the taxonomy gold examples / tie-breaks, and mark `review_note`.
That is the only way to catch labels that look legal but are wrong.
