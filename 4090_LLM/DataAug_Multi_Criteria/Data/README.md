# Data Directory

This directory hosts all datasets required for the multi-agent DSM-5 pipeline.

## Criteria Matching Data
- `groundtruth/redsm5_ground_truth.json`: JSONL file containing Reddit posts annotated with DSM-5 criteria.
- `DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json`: DSM-5 criteria definitions aligned with annotation IDs.

## Evidence Binding Data
- `evidence/annotations.jsonl`: Expected JSONL file where each record includes:
  - `post_id`: Identifier aligning with the criteria dataset.
  - `text`: Original Reddit post text.
  - `criterion_id`: DSM-5 criterion identifier (e.g., `A.1`).
  - `criteria_text`: Criterion description used as the query for span extraction.
  - `evidence_start` / `evidence_end`: Character offsets for the supporting evidence span. Use `-1` for examples without evidence.

Populate the evidence file with your dataset before running span-training or evaluation.
