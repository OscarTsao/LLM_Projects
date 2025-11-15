# Data Pipeline Implementation - STRICT Validation

## Overview

This document describes the core data pipeline implementation with **STRICT validation rules** that enforce field separation between criteria and evidence tasks.

## Critical Requirements (HIGHEST PRIORITY)

### 1. STRICT Field Mapping Rules

**These rules are enforced with assertions that FAIL if violated:**

- **Criteria labels** come **ONLY** from the `status` field in annotations
- **Evidence** comes **ONLY** from the `cases` field in annotations  
- **NO cross-contamination** is allowed between fields
- Any attempt to use the wrong field triggers an `AssertionError`

### 2. Field Mapping Behavior

#### Status Field (Criteria Task ONLY)
```python
# Normalized to binary labels
positive_values = ['positive', 'present', 'true', '1', 1, True] → 1
negative_values = ['negative', 'absent', 'false', '0', 0, False] → 0
invalid_values → dropped with warning
```

#### Cases Field (Evidence Task ONLY)
```python
# Parsed and exploded into individual evidence spans
cases_format = [
    {
        "text": "evidence text",
        "start_char": 0,
        "end_char": 13,
        "sentence_id": 0
    },
    ...
]
# Can be JSON string, list, or stringified list
# Empty cases [] are filtered out
```

## Implementation Files

### 1. `configs/data/field_map.yaml`

Comprehensive field mapping configuration with:
- Posts table column mappings
- Annotations table column mappings  
- Status value normalization rules
- Cases structure definition
- Validation settings (strict mode, fail conditions)

**Location:** `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/configs/data/field_map.yaml`

### 2. `src/psy_agents_noaug/data/groundtruth.py`

Core groundtruth generation functions:

#### Key Functions

```python
def _assert_field_usage(field_name: str, expected_field: str, operation: str):
    """Assert correct field is used - FAILS if violated"""
    
def create_criteria_groundtruth(
    annotations: pd.DataFrame,
    posts: pd.DataFrame,
    field_map: Dict[str, Any],
    valid_criterion_ids: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Create criteria groundtruth using ONLY status field.
    
    Returns DataFrame with columns:
    - post_id
    - criterion_id
    - status (original value)
    - label (normalized 0/1)
    
    Raises:
        AssertionError: If status field is not used
    """

def create_evidence_groundtruth(
    annotations: pd.DataFrame,
    posts: pd.DataFrame,
    field_map: Dict[str, Any],
    valid_criterion_ids: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Create evidence groundtruth using ONLY cases field.
    
    Returns DataFrame with columns:
    - post_id
    - criterion_id
    - case_id (index within post-criterion pair)
    - evidence_text
    - start_char
    - end_char
    - sentence_id
    
    Raises:
        AssertionError: If cases field is not used
    """

def validate_strict_separation(
    criteria_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    field_map: Dict[str, Any]
):
    """
    Validate that criteria and evidence use separate fields.
    
    Raises:
        AssertionError: If field separation is violated
    """
```

#### Key Classes

```python
class GroundTruthValidator:
    """Validates ground truth data against STRICT rules."""
    
    def validate_criteria_groundtruth(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Returns {'errors': [...], 'warnings': [...]}"""
    
    def validate_evidence_groundtruth(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Returns {'errors': [...], 'warnings': [...]}"""
```

**Location:** `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/src/psy_agents_noaug/data/groundtruth.py`

### 3. `src/psy_agents_noaug/data/loaders.py`

Data loading with strict validation:

#### Key Classes

```python
class ReDSM5DataLoader:
    """
    Loader for ReDSM-5 dataset with strict field validation.
    
    Can load from:
    - HuggingFace: 'irlab-udc/redsm5'
    - Local CSVs: posts.csv and annotations.csv
    """
    
    def __init__(
        self,
        field_map: Dict[str, Any],
        data_source: str = 'local',
        data_dir: Optional[Path] = None,
        hf_dataset_name: Optional[str] = None
    ):
        """Initialize loader with field mapping."""
    
    def load_posts(self) -> pd.DataFrame:
        """Load posts with validation."""
    
    def load_annotations(self) -> pd.DataFrame:
        """Load annotations with validation."""
    
    def load_dsm_criteria(self, dsm_path: Path) -> List[Dict[str, str]]:
        """Load DSM criteria JSON."""
    
    def get_valid_criterion_ids(self, dsm_path: Path) -> Set[str]:
        """Get set of valid criterion IDs."""

class DSMCriteriaLoader:
    """Loader for DSM-5 criteria knowledge base."""
    
    def load_criteria(self) -> List[Dict[str, str]]:
        """Load all criteria."""
    
    def get_criterion_by_id(self, criterion_id: str) -> Optional[Dict[str, str]]:
        """Get specific criterion."""
    
    def get_all_criterion_ids(self) -> List[str]:
        """Get all criterion IDs."""
```

#### Key Functions

```python
def group_split_by_post_id(
    df: pd.DataFrame,
    post_id_col: str = 'post_id',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split data by post_id to prevent data leakage.
    
    All annotations for a post_id stay in same split.
    Returns (train_post_ids, val_post_ids, test_post_ids)
    """

def save_splits_json(
    train_post_ids: List[str],
    val_post_ids: List[str],
    test_post_ids: List[str],
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save splits to JSON for reproducibility."""

def load_splits_json(splits_path: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Load splits from JSON.
    
    Validates no data leakage (no overlapping post_ids).
    """
```

**Location:** `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/src/psy_agents_noaug/data/loaders.py`

### 4. `scripts/make_groundtruth.py`

Command-line script to generate ground truth datasets:

```bash
# Usage examples:

# Basic usage (local CSV files)
python scripts/make_groundtruth.py

# Specify custom paths
python scripts/make_groundtruth.py \
    --data-dir /path/to/data \
    --output-dir /path/to/output \
    --dsm-criteria /path/to/dsm_criteria.json

# Use HuggingFace dataset
python scripts/make_groundtruth.py \
    --data-source huggingface \
    --hf-dataset irlab-udc/redsm5

# Custom split ratios
python scripts/make_groundtruth.py \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --random-seed 42
```

#### Output Files

1. **`data/processed/criteria_groundtruth.csv`**
   ```
   post_id,criterion_id,status,label
   post1,A,positive,1
   post2,B,negative,0
   ...
   ```

2. **`data/processed/evidence_groundtruth.csv`**
   ```
   post_id,criterion_id,case_id,evidence_text,start_char,end_char,sentence_id
   post1,A,0,"evidence text",5,18,0
   post1,A,1,"another evidence",25,40,1
   ...
   ```

3. **`data/processed/splits.json`**
   ```json
   {
     "train": ["post1", "post2", ...],
     "val": ["post10", "post11", ...],
     "test": ["post20", "post21", ...],
     "metadata": {
       "random_seed": 42,
       "train_ratio": 0.7,
       "val_ratio": 0.15,
       "test_ratio": 0.15,
       "train_count": 100,
       "val_count": 21,
       "test_count": 22
     }
   }
   ```

**Location:** `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/scripts/make_groundtruth.py`

## Data Integrity Checks

The pipeline performs these validation checks:

### 1. Required Column Validation
- Posts must have: `post_id`, `text`
- Annotations must have: `post_id`, `criterion_id`, `status`, `cases`
- Missing columns trigger `ValueError`

### 2. Criterion ID Validation
- All criterion IDs must exist in DSM criteria JSON
- Invalid IDs trigger `ValueError` (configurable)

### 3. Post ID Validation
- All annotation post_ids must exist in posts table
- Missing post_ids trigger `ValueError` (configurable)

### 4. Duplicate Handling
- Exact duplicates are dropped with warning
- Conflicts (same post_id + criterion_id, different labels) are warned

### 5. Data Leakage Prevention
- Splits are created by grouping on post_id
- All annotations for a post stay in same split
- Loading splits validates no overlap between train/val/test

### 6. Deterministic Splits
- Same random seed produces same splits
- Splits are saved to JSON for reproducibility

## Testing

### Test Files

1. **`tests/test_groundtruth.py`**
   - Tests strict field usage assertions
   - Tests criteria/evidence groundtruth creation
   - Tests validation functions
   - Tests split determinism and no data leakage

2. **`tests/test_loaders.py`**
   - Tests data loader initialization
   - Tests CSV and HuggingFace loading
   - Tests split creation and saving/loading
   - Tests data leakage detection

**Locations:**
- `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/tests/test_groundtruth.py`
- `/experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence/tests/test_loaders.py`

### Running Tests

```bash
# Run all tests
cd /experiment/YuNing/Refactored_Psy/NoAug_Criteria_Evidence
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_groundtruth.py -v

# Run with coverage
python -m pytest tests/ --cov=psy_agents_noaug.data
```

## Usage Example

```python
import sys
from pathlib import Path

# Add src to path if needed
sys.path.insert(0, 'src')

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

# Load field mapping
field_map = load_field_map('configs/data/field_map.yaml')

# Initialize loader
loader = ReDSM5DataLoader(
    field_map=field_map,
    data_source='local',
    data_dir=Path('data/raw/redsm5')
)

# Load data
posts = loader.load_posts()
annotations = loader.load_annotations()
valid_criterion_ids = loader.get_valid_criterion_ids('data/raw/redsm5/dsm_criteria.json')

# Create criteria groundtruth (uses ONLY status field)
criteria_gt = create_criteria_groundtruth(
    annotations=annotations,
    posts=posts,
    field_map=field_map,
    valid_criterion_ids=valid_criterion_ids
)

# Create evidence groundtruth (uses ONLY cases field)
evidence_gt = create_evidence_groundtruth(
    annotations=annotations,
    posts=posts,
    field_map=field_map,
    valid_criterion_ids=valid_criterion_ids
)

# Validate strict separation
validate_strict_separation(criteria_gt, evidence_gt, field_map)

# Create splits with no data leakage
train_ids, val_ids, test_ids = group_split_by_post_id(
    df=annotations,
    post_id_col='post_id',
    random_seed=42
)

# Save splits
save_splits_json(
    train_post_ids=train_ids,
    val_post_ids=val_ids,
    test_post_ids=test_ids,
    output_path='data/processed/splits.json'
)
```

## Verification

### Verify Strict Rules Are Enforced

```python
# This should PASS
criteria_gt = create_criteria_groundtruth(...)
assert 'status' in criteria_gt.columns
assert 'cases' not in criteria_gt.columns
assert 'evidence_text' not in criteria_gt.columns

# This should PASS
evidence_gt = create_evidence_groundtruth(...)
assert 'evidence_text' in evidence_gt.columns
assert 'status' not in evidence_gt.columns
assert 'label' not in evidence_gt.columns

# This should PASS
validate_strict_separation(criteria_gt, evidence_gt, field_map)
# Prints: "VALIDATION PASSED: Strict field separation maintained"
```

### Verify No Data Leakage

```python
from psy_agents_noaug.data.loaders import load_splits_json

# This validates no overlap during loading
train_ids, val_ids, test_ids = load_splits_json('data/processed/splits.json')

# Manual verification
train_set = set(train_ids)
val_set = set(val_ids)
test_set = set(test_ids)

assert len(train_set & val_set) == 0  # No overlap
assert len(train_set & test_set) == 0  # No overlap
assert len(val_set & test_set) == 0  # No overlap
```

## DataAug Repository

**The same implementation exists in the DataAug repository** with identical validation rules:

- Location: `/experiment/YuNing/Refactored_Psy/DataAug_Criteria_Evidence/`
- Package: `psy_agents_aug` (instead of `psy_agents_noaug`)
- All files, tests, and validation rules are identical
- Ensures consistency between NoAug and DataAug pipelines

## Summary

### Files Implemented

1. `configs/data/field_map.yaml` - Field mapping configuration
2. `src/psy_agents_noaug/data/groundtruth.py` - Groundtruth generation with strict validation
3. `src/psy_agents_noaug/data/loaders.py` - Data loaders with validation
4. `scripts/make_groundtruth.py` - CLI script for groundtruth generation
5. `tests/test_groundtruth.py` - Tests for groundtruth functions
6. `tests/test_loaders.py` - Tests for data loaders

### STRICT Rules Enforced

✓ Criteria labels from 'status' field ONLY  
✓ Evidence from 'cases' field ONLY  
✓ Assertions FAIL if wrong fields are used  
✓ No cross-contamination between tasks  
✓ No data leakage in splits (post_id grouping)  
✓ Deterministic splits with seed  
✓ Comprehensive validation and integrity checks  

### Verification Status

✓ All implementation files created in both repositories  
✓ All tests pass successfully  
✓ STRICT validation rules enforced with assertions  
✓ No data leakage verified  
✓ Documentation complete  
