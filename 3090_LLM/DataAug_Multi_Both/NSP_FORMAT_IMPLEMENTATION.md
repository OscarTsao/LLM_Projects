# Next Sentence Prediction (NSP) Format Implementation

## Summary

Modified the input format to use Next Sentence Prediction (NSP) style tokenization for both binary pairs and multi-label modes. This allows the model to better understand the relationship between posts and mental health criteria.

## Changes Made

### 1. Dataset Module (`src/dataaug_multi_both/data/dataset.py`)

#### Added DSM-5 Criterion Descriptions
- Added `CRITERION_TEXTS` class variable with human-readable descriptions for all 9 DSM-5 criteria:
  - SLEEP_ISSUES: "Sleep issues or insomnia"
  - ANHEDONIA: "Loss of interest or anhedonia"
  - APPETITE_CHANGE: "Appetite or weight change"
  - FATIGUE: "Fatigue or loss of energy"
  - WORTHLESSNESS: "Worthlessness or guilt"
  - COGNITIVE_ISSUES: "Cognitive or concentration issues"
  - PSYCHOMOTOR: "Psychomotor agitation or retardation"
  - SUICIDAL_THOUGHTS: "Suicidal thoughts or ideation"
  - DEPRESSED_MOOD: "Depressed mood"

#### Modified Tokenization in `__getitem__` Method

**Binary Pairs Format:**
- **Old:** `[CLS] post [SEP]`
- **New:** `[CLS] post [SEP] criterion [SEP]`
- Uses sentence pair tokenization: `tokenizer(post_text, criterion_text, ...)`
- Each post generates 9 examples (one per criterion)

**Multi-Label Format:**
- **Old:** `[CLS] post [SEP]`
- **New:** `[CLS] post [SEP] criterion1 criterion2 ... criterion9 [SEP]`
- Uses sentence pair tokenization: `tokenizer(post_text, all_criteria, ...)`
- All criteria concatenated with spaces
- One example per post

#### Added Token Type IDs
- Now includes `token_type_ids` in the output
- `token_type_ids = 0` for post tokens
- `token_type_ids = 1` for criterion tokens
- Enables the model to distinguish between the two segments

### 2. Multi-Task Model (`src/dataaug_multi_both/models/multi_task_model.py`)

#### Updated `forward()` Method
- Added `token_type_ids` parameter (optional)
- Passes `token_type_ids` to encoder when available
- Maintains backward compatibility (works without token_type_ids)

```python
def forward(
    self,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,  # NEW
    return_encoder_outputs: bool = False,
    return_predictions: bool = False,
    criteria_threshold: float = 0.5
) -> MultiTaskModelOutput:
```

### 3. Training Loop (`src/dataaug_multi_both/hpo/trial_executor.py`)

#### Updated Training Step
- Modified forward pass to include `token_type_ids` from batch
- Checks if `token_type_ids` exists in batch before passing to model

**Training loop:**
```python
model_inputs = {
    "input_ids": batch["input_ids"],
    "attention_mask": batch["attention_mask"]
}
if "token_type_ids" in batch:
    model_inputs["token_type_ids"] = batch["token_type_ids"]

outputs = model(**model_inputs)
```

#### Updated Validation Step
- Same pattern applied to validation loop
- Maintains consistency with training

## Format Examples

### Binary Pairs Format
```
Post: "I can't sleep at night and feel tired all day"

Example 0 (SLEEP_ISSUES):
Input: [CLS] i can't sleep at night and feel tired all day [SEP] Sleep issues or insomnia [SEP]
Token Type IDs: [0, 0, 0, ..., 0, 1, 1, 1, 1, 1, 1, 0, 0, ...]
Label: [1, 0, 0, 0, 0, 0, 0, 0, 0]

Example 1 (ANHEDONIA):
Input: [CLS] i can't sleep at night and feel tired all day [SEP] Loss of interest or anhedonia [SEP]
Token Type IDs: [0, 0, 0, ..., 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, ...]
Label: [0, 0, 0, 0, 0, 0, 0, 0, 0]

... (9 examples total)
```

### Multi-Label Format
```
Post: "I can't sleep at night and feel tired all day"

Input: [CLS] i can't sleep at night and feel tired all day [SEP] Sleep issues or insomnia Loss of interest or anhedonia Appetite or weight change Fatigue or loss of energy ... Depressed mood [SEP]
Token Type IDs: [0, 0, 0, ..., 0, 1, 1, 1, 1, 1, ... 1, 1, 0, 0, ...]
Labels: [1, 0, 0, 1, 0, 0, 0, 0, 0]
```

## Benefits

1. **Better Semantic Understanding**: The model can learn relationships between posts and specific criteria
2. **NSP Pre-training Alignment**: Leverages BERT's pre-trained NSP objective
3. **Explicit Criterion Context**: Each criterion is explicitly presented to the model
4. **Token Type Discrimination**: Model can distinguish post from criteria via token_type_ids
5. **Backward Compatible**: Works with models that don't use token_type_ids

## Testing

All changes have been tested:
- ✓ Dataset tokenization produces correct format
- ✓ Token type IDs are properly generated
- ✓ Model accepts and processes token_type_ids
- ✓ Training loop correctly passes all inputs
- ✓ Both binary_pairs and multi_label formats work correctly

## Migration Notes

- No configuration changes needed
- Existing training scripts will work as-is
- Token type IDs are automatically included when tokenizer supports them
- Models trained with old format can be retrained with new format
