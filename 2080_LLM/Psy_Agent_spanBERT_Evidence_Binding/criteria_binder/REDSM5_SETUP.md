# RedSM5 Dataset Integration

This document explains how the codebase has been updated to work with the renewed RedSM5 dataset structure.

## Dataset Structure

The new data structure consists of:

1. **DSM-5 Criteria**: `Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json`

   - Contains the official DSM-5 criteria for Major Depressive Disorder
   - Each criterion has an ID (A.1, A.2, etc.) and descriptive text

2. **Posts**: `Data/redsm5/redsm5_posts.csv`

   - Contains post_id and text columns
   - Full text of social media posts

3. **Annotations**: `Data/redsm5/redsm5_annotations.csv`
   - Contains post_id, sentence_id, sentence_text, DSM5_symptom, status, explanation
   - sentence_text: Evidence sentences that support/contradict criteria
   - DSM5_symptom: Which DSM-5 criterion the evidence relates to
   - status: 1 if criterion is matched, 0 if not matched

## Code Changes

### 1. New Data Loader (`src/data/redsm5_loader.py`)

Created a comprehensive data loader that:

- Loads DSM-5 criteria and maps symptom codes to criterion text
- Combines posts with annotations to create training examples
- Handles evidence span extraction by finding sentence_text within full post text
- Supports train/dev/test splitting by post_id to avoid data leakage

### 2. Data Preparation Script (`scripts/prepare_redsm5_data.py`)

Automated script that:

- Processes the raw data files
- Generates JSONL files in the format expected by the existing model
- Splits data into train/dev/test sets (70/15/15 by default)
- Creates balanced splits while avoiding data leakage

### 3. Updated Configuration (`src/config/`)

Updated data paths live in the Hydra config defaults. The training config composes the following:

```bash
python -m src.cli train --config-path src/config --config-name train --cfg job
```

Relevant entries:

```yaml
# src/config/data/redsm5.yaml
data:
  train_path: "data/redsm5/train.jsonl"
  dev_path: "data/redsm5/dev.jsonl"
  test_path: "data/redsm5/test.jsonl"
```

## Symptom Mapping

The loader maps DSM5_symptom codes to DSM-5 criteria:

| Symptom Code      | DSM-5 Criterion | Description                         |
| ----------------- | --------------- | ----------------------------------- |
| DEPRESSED_MOOD    | A.1             | Depressed mood most of the day      |
| ANHEDONIA         | A.2             | Diminished interest or pleasure     |
| APPETITE_CHANGE   | A.3             | Significant weight/appetite changes |
| SLEEP_ISSUES      | A.4             | Insomnia or hypersomnia             |
| PSYCHOMOTOR       | A.5             | Psychomotor agitation/retardation   |
| FATIGUE           | A.6             | Fatigue or loss of energy           |
| WORTHLESSNESS     | A.7             | Feelings of worthlessness/guilt     |
| COGNITIVE_ISSUES  | A.8             | Diminished thinking/concentration   |
| SUICIDAL_THOUGHTS | A.9             | Recurrent thoughts of death/suicide |
| SPECIAL_CASE      | A.1             | Mapped to depressed mood by default |

## Generated Dataset Statistics

After processing:

- **Total examples**: 2,058
- **Training set**: 1,433 examples (1,143 positive, 290 negative)
- **Dev set**: 320 examples (253 positive, 67 negative)
- **Test set**: 305 examples (235 positive, 70 negative)
- **Unique posts**: 1,477
- **Unique symptoms**: 10

## Usage

### 1. Prepare Data

```bash
python scripts/prepare_redsm5_data.py
```

### 2. Train Model

```bash
python -m src.cli train --config-path src/config --config-name train
```

### 3. Custom Parameters

```bash
python -m src.cli train --config-path src/config --config-name train \
  train.batch_size=4 model.max_length=256 train.epochs=10
```

## Data Format

Each training example contains:

- `id`: Unique identifier combining post_id, symptom, and sentence_id
- `criterion_text`: DSM-5 criterion text
- `document_text`: Full post text
- `label`: 1 if criterion matched, 0 if not matched
- `evidence_char_spans`: Character spans of evidence sentences in the post
- `post_id`, `sentence_id`, `dsm5_symptom`, `sentence_text`: Metadata

## Validation

The system has been tested and validates:

- ✅ Data loading works with existing CriteriaBindingDataset
- ✅ Tokenization and alignment functions properly
- ✅ Training pipeline starts successfully
- ✅ Model can process the evidence spans correctly

## Memory Considerations

For training on limited GPU memory:

- Reduce batch_size (e.g., --overrides train.batch_size=4)
- Reduce max_length (e.g., --overrides model.max_length=256)
- Use gradient accumulation if needed
