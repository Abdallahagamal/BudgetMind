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

## Dataset Statistics
Raw tasks: 353
After removing duplicates: 
After removing invalid records: 

## Fields
task_id
task_text
answer
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

## Labeling
TYPE, COMPLEXITY, and DOMAIN are not assigned yet.
They will be assigned by the labeling team according
to the Sprint 1 taxonomy.