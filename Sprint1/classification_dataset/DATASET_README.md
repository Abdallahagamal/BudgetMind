# BudgetMind Classification Dataset

## Purpose
This dataset contains raw user tasks that will be
classified by the BudgetMind Task Classification Layer.

## Sources
- TruthfulQA
- Natural Question(NQ)
- WMT translation
- cnn_dailymail
- Xsum 
- google-research-datasets
- SWE-bench
- openai/gsm8k
- mmlu
- BIG-Bench-Hard
- devpkprajapati/ai-system-design-instruct
- HLD-Bench
- Hugging Face Creative Writing datasets


## Fields
task_id
task_text
source
date_collected
type
complexity
domain

## Cleaning
- Removed empty records
- Removed invalid/non-task records
- Removed duplicate tasks
- Normalized whitespace
- Preserved technical terminology


### What Changed

| Fix | Rows Affected | What Was Wrong |
|---|---:|---|
| Fixed doubled quotes | 18 (`cnn_dailymail`) | `""` appeared where `"` was intended due to a CSV-escaping artifact left over from scraping. |
| Normalized line endings | 21 of 38 (`SWE-bench`) | Windows-style `\r\n` line endings appeared inside embedded code. They were normalized to consistent `\n` line endings. |
| Collapsed double spaces | 7 (`GSM8K`) | Double spaces after periods were removed. This was a formatting inconsistency found in the GSM8K records. |
| Fixed casing and punctuation | 35 (Natural Questions) | Raw lowercase search queries without punctuation (e.g., `"where did they film..."`) were converted into properly formatted questions. |
| Exact duplicates | 0 | No exact duplicate records were found. |

### What I Flagged Instead of Deleting

A total of **73 candidate near-duplicate pairs** were identified using TF-IDF similarity. However, most were **false positives rather than actual duplicates**, so they were retained.

- **WMT rows:** The same source sentence may appear with different target languages, such as Finnish, German, French, or Spanish. These rows are lexically similar because the English source content is repeated, but they represent genuinely different translation tasks. Therefore, they were kept.

- **`devpkprajapati` architecture rows:** This was the main finding. The same system concept (for example, **"Real-Time Fraud Scoring Service"**) appears 3–4 times with different QPS and latency requirements, such as 35,000 QPS vs. 9,000 QPS. These are not exact duplicates, and the different scale requirements could reasonably affect the **COMPLEXITY** label, potentially changing it from Medium to High. Therefore, they were retained.

  However, this means that the **60 architecture-design rows represent approximately 16 distinct system concepts**, rather than 60 completely different concepts. This should be a team decision: either keep the variants because the different constraints create meaningful task differences, or limit the dataset to a maximum of 2 variants per system concept and replace the remaining rows with more diverse architecture prompts.


## Dataset Statistics
Raw tasks: 540
After removing duplicates: 540
After removing invalid records: 540

## Labeling
TYPE, COMPLEXITY, and DOMAIN are not assigned yet.
They will be assigned by the labeling team according
to the Sprint 1 taxonomy.


## Notes for Sprint 1 to be done (Esraa)
 1. Tasks "T0461, T0467, T0427, T0415, T0301,T0305,T0313,T0433,T0438, T0297,T0270,T0261,T0116,T0018" didn't get labeled
 2. converting "labeled_dataset.csv" into "labeled_dataset.jsonl"
 3. Havn't done "split Train/Val/Test" task yet
 4. we need to Run generate_embeddings.py on the real splits