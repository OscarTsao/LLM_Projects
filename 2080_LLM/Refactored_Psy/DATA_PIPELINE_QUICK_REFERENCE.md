# Data Pipeline Quick Reference

## STRICT Validation Rules (HIGHEST PRIORITY)

```python
# RULE 1: Criteria labels come ONLY from 'status' field
criteria_gt = create_criteria_groundtruth(annotations, posts, field_map, valid_ids)
# → Returns: post_id, criterion_id, status, label
# → NO 'cases' or 'evidence_text' columns allowed

# RULE 2: Evidence comes ONLY from 'cases' field
evidence_gt = create_evidence_groundtruth(annotations, posts, field_map, valid_ids)
# → Returns: post_id, criterion_id, case_id, evidence_text, start_char, end_char, sentence_id
# → NO 'status' or 'label' columns allowed

# RULE 3: Validation enforces separation
validate_strict_separation(criteria_gt, evidence_gt, field_map)
# → Raises AssertionError if rules violated
```

## Generate Ground Truth (Command Line)

```bash
# Basic usage
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence
python scripts/make_groundtruth.py

# With custom paths
python scripts/make_groundtruth.py \
    --data-dir /path/to/data \
    --output-dir /path/to/output \
    --random-seed 42

# From HuggingFace
python scripts/make_groundtruth.py \
    --data-source huggingface \
    --hf-dataset irlab-udc/redsm5
```

## Output Files

```
data/processed/
├── criteria_groundtruth.csv
│   ├── post_id
│   ├── criterion_id
│   ├── status (original value)
│   └── label (0 or 1)
│
├── evidence_groundtruth.csv
│   ├── post_id
│   ├── criterion_id
│   ├── case_id
│   ├── evidence_text
│   ├── start_char
│   ├── end_char
│   └── sentence_id
│
└── splits.json
    ├── train: [post_ids...]
    ├── val: [post_ids...]
    ├── test: [post_ids...]
    └── metadata: {seed, ratios, counts}
```

## Programmatic Usage

```python
import sys
from pathlib import Path

# Add to path
sys.path.insert(0, 'src')

# Import functions
from psy_agents_noaug.data.groundtruth import (
    create_criteria_groundtruth,
    create_evidence_groundtruth,
    validate_strict_separation,
    load_field_map,
)
from psy_agents_noaug.data.loaders import (
    ReDSM5DataLoader,
    group_split_by_post_id,
    save_splits_json,
)

# 1. Load field mapping
field_map = load_field_map('configs/data/field_map.yaml')

# 2. Initialize loader
loader = ReDSM5DataLoader(
    field_map=field_map,
    data_source='local',
    data_dir=Path('data/raw/redsm5')
)

# 3. Load data
posts = loader.load_posts()
annotations = loader.load_annotations()
valid_ids = loader.get_valid_criterion_ids('data/raw/redsm5/dsm_criteria.json')

# 4. Create groundtruth (USES CORRECT FIELDS)
criteria_gt = create_criteria_groundtruth(
    annotations, posts, field_map, valid_ids
)  # Uses 'status' field ONLY

evidence_gt = create_evidence_groundtruth(
    annotations, posts, field_map, valid_ids
)  # Uses 'cases' field ONLY

# 5. Validate separation (ENFORCES STRICT RULES)
validate_strict_separation(criteria_gt, evidence_gt, field_map)

# 6. Create splits (NO DATA LEAKAGE)
train_ids, val_ids, test_ids = group_split_by_post_id(
    df=annotations,
    post_id_col='post_id',
    random_seed=42
)

# 7. Save splits (REPRODUCIBILITY)
save_splits_json(train_ids, val_ids, test_ids, 'data/processed/splits.json')
```

## Status Value Mapping

```yaml
# In field_map.yaml
status_values:
  positive: ["positive", "present", "true", "1", 1, True]  → 1
  negative: ["negative", "absent", "false", "0", 0, False] → 0
  # Other values dropped with warning
```

## Cases Field Structure

```python
# Input (can be JSON string, list, or stringified list)
cases = '[{"text": "evidence", "start_char": 0, "end_char": 8, "sentence_id": 0}]'

# Output after parsing and exploding
evidence_gt:
  post_id, criterion_id, case_id, evidence_text, start_char, end_char, sentence_id
  post1,   A,            0,       "evidence",    0,          8,        0
```

## Data Leakage Prevention

```python
# CORRECT: Split by post_id (all annotations for a post stay together)
train_ids, val_ids, test_ids = group_split_by_post_id(
    df=annotations,
    post_id_col='post_id',
    random_seed=42
)

# Verification (automatically done by load_splits_json)
train_set = set(train_ids)
val_set = set(val_ids)
test_set = set(test_ids)
assert len(train_set & val_set) == 0  # No overlap
assert len(train_set & test_set) == 0  # No overlap
assert len(val_set & test_set) == 0   # No overlap
```

## Validation Checks

```python
# 1. Field usage (automatic)
# → AssertionError if wrong field used

# 2. Criterion IDs (automatic)
# → ValueError if invalid ID (configurable)

# 3. Post IDs (automatic)
# → ValueError if post not found (configurable)

# 4. Duplicates (automatic)
# → Warning and drop

# 5. Data leakage (automatic)
# → ValueError on load if overlap detected
```

## Common Use Cases

### 1. Generate Ground Truth from Local Files

```bash
python scripts/make_groundtruth.py \
    --data-dir data/raw/redsm5 \
    --output-dir data/processed
```

### 2. Use Custom Split Ratios

```bash
python scripts/make_groundtruth.py \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1
```

### 3. Load Existing Splits

```python
from psy_agents_noaug.data.loaders import load_splits_json

train_ids, val_ids, test_ids = load_splits_json('data/processed/splits.json')
```

### 4. Validate Custom Data

```python
from psy_agents_noaug.data.groundtruth import GroundTruthValidator

validator = GroundTruthValidator(field_map, valid_criterion_ids)

# Validate criteria
result = validator.validate_criteria_groundtruth(criteria_df)
if result['errors']:
    print("Errors:", result['errors'])

# Validate evidence
result = validator.validate_evidence_groundtruth(evidence_df)
if result['errors']:
    print("Errors:", result['errors'])
```

## File Locations

### NoAug Repository
```
/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/
├── configs/data/field_map.yaml
├── src/psy_agents_noaug/data/
│   ├── groundtruth.py
│   ├── loaders.py
│   └── __init__.py
├── scripts/make_groundtruth.py
└── tests/
    ├── test_groundtruth.py
    └── test_loaders.py
```

### DataAug Repository
```
/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/
├── configs/data/field_map.yaml
├── src/psy_agents_aug/data/
│   ├── groundtruth.py
│   ├── loaders.py
│   └── __init__.py
├── scripts/make_groundtruth.py
└── tests/
    ├── test_groundtruth.py
    └── test_loaders.py
```

## Testing

```bash
# Run all tests
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence
python -m pytest tests/ -v

# Run specific test
python -m pytest tests/test_groundtruth.py::test_create_criteria_groundtruth -v

# Run with output
python -m pytest tests/ -v -s
```

## Troubleshooting

### Import Error
```python
# Add src to path
import sys
sys.path.insert(0, 'src')
```

### Missing PyYAML
```bash
pip install pyyaml
```

### Validation Failure
```python
# Check field_map.yaml has correct mappings
# Ensure status field is 'status' (not something else)
# Ensure cases field is 'cases' (not something else)
```

### Data Leakage Detected
```python
# Regenerate splits with group_split_by_post_id
# Never split annotations directly - always split post_ids
```

## Key Assertions

```python
# These will FAIL if violated:

# 1. Criteria must use status field
assert field_name == 'status', "Criteria must use status field"

# 2. Evidence must use cases field
assert field_name == 'cases', "Evidence must use cases field"

# 3. No contamination in criteria
assert 'cases' not in criteria_df.columns
assert 'evidence_text' not in criteria_df.columns

# 4. No contamination in evidence
assert 'status' not in evidence_df.columns
assert 'label' not in evidence_df.columns

# 5. No data leakage in splits
assert len(train_set & val_set) == 0
assert len(train_set & test_set) == 0
assert len(val_set & test_set) == 0
```

## Summary

✓ Criteria: Use `status` field ONLY → (post_id, criterion_id, status, label)  
✓ Evidence: Use `cases` field ONLY → (post_id, criterion_id, case_id, evidence_text, spans)  
✓ Validation: Enforced with assertions that FAIL if violated  
✓ Splits: Group by post_id to prevent data leakage  
✓ Reproducibility: Save splits to JSON with seed  
