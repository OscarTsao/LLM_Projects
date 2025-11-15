# NSP Format Implementation - Complete ✓

## What Was Changed

Successfully modified the input format to use **Next Sentence Prediction (NSP)** style, where the post and criteria are provided as sentence pairs to the model.

## Implementation Details

### 1. **Dataset Changes** (`src/dataaug_multi_both/data/dataset.py`)

#### Added DSM-5 Criterion Text Descriptions
```python
CRITERION_TEXTS = [
    "Sleep issues or insomnia",           # 0: SLEEP_ISSUES
    "Loss of interest or anhedonia",      # 1: ANHEDONIA
    "Appetite or weight change",          # 2: APPETITE_CHANGE
    "Fatigue or loss of energy",          # 3: FATIGUE
    "Worthlessness or guilt",             # 4: WORTHLESSNESS
    "Cognitive or concentration issues",  # 5: COGNITIVE_ISSUES
    "Psychomotor agitation or retardation", # 6: PSYCHOMOTOR
    "Suicidal thoughts or ideation",      # 7: SUICIDAL_THOUGHTS
    "Depressed mood",                     # 8: DEPRESSED_MOOD
]
```

#### Modified Input Format

**Binary Pairs Mode:**
- **Before:** `[CLS] post [SEP]`
- **After:** `[CLS] post [SEP] criterion_i [SEP]`
- Creates 9 examples per post (one for each criterion)
- Uses `tokenizer(post, criterion_text)` for proper NSP format

**Multi-Label Mode:**
- **Before:** `[CLS] post [SEP]`
- **After:** `[CLS] post [SEP] criterion1 criterion2 ... criterion9 [SEP]`
- Creates 1 example per post with all criteria
- Uses `tokenizer(post, all_criteria_concatenated)` for NSP format

#### Added Token Type IDs
- Returns `token_type_ids` in dataset output
- `0` for post tokens, `1` for criterion tokens
- Enables model to distinguish segments

### 2. **Model Changes** (`src/dataaug_multi_both/models/multi_task_model.py`)

Updated `forward()` method signature:
```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,  # NEW
    ...
) -> MultiTaskModelOutput:
```

Passes `token_type_ids` to encoder when available.

### 3. **Training Changes** (`src/dataaug_multi_both/hpo/trial_executor.py`)

Updated both training and validation loops:
```python
model_inputs = {
    "input_ids": batch["input_ids"],
    "attention_mask": batch["attention_mask"]
}
if "token_type_ids" in batch:
    model_inputs["token_type_ids"] = batch["token_type_ids"]

outputs = model(**model_inputs)
```

## Example Output

### Binary Pairs Format
```
Input: [CLS] i can't sleep at night [SEP] Sleep issues or insomnia [SEP]
Token Type IDs: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, ...]
Label for this criterion: [1, 0, 0, 0, 0, 0, 0, 0, 0]
```

### Multi-Label Format
```
Input: [CLS] i can't sleep at night [SEP] Sleep issues or insomnia Loss of interest ... Depressed mood [SEP]
Token Type IDs: [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, ..., 1, 0, 0, ...]
Labels for all criteria: [1, 0, 0, 1, 0, 0, 0, 0, 0]
```

## Benefits

1. **Semantic Relationship**: Model can learn explicit relationships between posts and criteria
2. **Pre-training Alignment**: Leverages BERT's NSP pre-training objective
3. **Explicit Context**: Each criterion is explicitly presented as context
4. **Segment Discrimination**: Token type IDs help model distinguish post from criteria
5. **Backward Compatible**: Works with or without token_type_ids

## Files Modified

1. `src/dataaug_multi_both/data/dataset.py` - Dataset tokenization
2. `src/dataaug_multi_both/models/multi_task_model.py` - Model forward pass
3. `src/dataaug_multi_both/hpo/trial_executor.py` - Training/validation loops

## Testing

All changes verified with:
- ✓ Syntax compilation check
- ✓ Dataset tokenization produces correct format
- ✓ Token type IDs properly generated
- ✓ Model accepts token_type_ids
- ✓ End-to-end forward pass works
- ✓ Both binary_pairs and multi_label modes functional

## Usage

No configuration changes needed. The new format is automatically applied when using the dataset:

```python
dataset = RedSM5Dataset(
    hf_dataset=hf_dataset,
    tokenizer=tokenizer,
    input_format="binary_pairs",  # or "multi_label"
    max_length=512
)
```

## Verification

Run the verification script to see the NSP format in action:
```bash
python verify_nsp_format.py
```

---
**Status**: ✓ Implementation Complete and Tested
**Date**: 2025
