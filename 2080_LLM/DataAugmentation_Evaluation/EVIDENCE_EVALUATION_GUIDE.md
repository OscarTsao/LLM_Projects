# Evidence Binding Agent - Detailed Evaluation Guide

This guide explains how to run comprehensive evaluation for the evidence binding agent, which will generate detailed predictions and metrics for both validation and test sets.

## Quick Start

### Using Make (Recommended)

```bash
make evaluate-evidence-detailed
```

This will:
1. Load the trained model from `outputs/train_Evidence/best/model.pt`
2. Evaluate on both **validation** and **test** sets
3. Save detailed predictions and metrics to `outputs/train_Evidence/evaluation/`

### Using Python Directly

```bash
PYTHONPATH=/experiment/YuNing/DataAugmentation_Evaluation python scripts/evaluate_evidence_detailed.py \
    --checkpoint_dir outputs/train_Evidence \
    --output_dir outputs/train_Evidence/evaluation
```

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint_dir` | `outputs/train_Evidence` | Directory containing model checkpoint and config |
| `--output_dir` | `outputs/train_Evidence/evaluation` | Directory to save evaluation results |
| `--batch_size` | `16` | Batch size for evaluation |
| `--model_name` | `microsoft/deberta-v3-base` | Model name for tokenizer |
| `--max_seq_length` | `512` | Maximum sequence length |
| `--span_threshold` | `0.5` | Threshold for span prediction |
| `--posts_path` | `Data/ReDSM5/redsm5_posts.csv` | Path to posts CSV |
| `--annotations_path` | `Data/ReDSM5/redsm5_annotations.csv` | Path to annotations CSV |

### Custom Example

```bash
PYTHONPATH=/experiment/YuNing/DataAugmentation_Evaluation python scripts/evaluate_evidence_detailed.py \
    --checkpoint_dir outputs/train_Evidence \
    --output_dir outputs/train_Evidence/evaluation_custom \
    --batch_size 32 \
    --span_threshold 0.6
```

## Output Files

The evaluation generates three main files in the `output_dir`:

### 1. `val_predictions.csv`

Detailed predictions for the validation set with the following columns:

| Column | Description |
|--------|-------------|
| `post_id` | Unique identifier for the post |
| `post` | Full text of the post |
| `criterion` | DSM-5 criterion ID |
| `has_evidence` | Whether ground truth has evidence (True/False) |
| `num_predicted_spans` | Number of predicted evidence spans |
| `num_ground_truth_spans` | Number of ground truth evidence spans |
| `predicted_spans` | Token positions of predicted spans (e.g., `[(10, 25), (30, 45)]`) |
| `ground_truth_spans` | Token positions of ground truth spans |
| `predicted_sentences` | Predicted evidence text (separated by ` \| `) |
| `ground_truth_sentences` | Ground truth evidence text (separated by ` \| `) |

**Example CSV Row:**
```csv
post_id,post,criterion,has_evidence,num_predicted_spans,num_ground_truth_spans,predicted_spans,ground_truth_spans,predicted_sentences,ground_truth_sentences
P001,"I feel sad all the time...",MDD_1,True,1,1,"[(15, 28)]","[(15, 28)]","feel sad all the time","feel sad all the time"
```

### 2. `test_predictions.csv`

Same format as validation predictions but for the test set.

### 3. `evaluation_metrics.json`

Comprehensive evaluation metrics in JSON format:

```json
{
  "val_start_accuracy": 0.9234,
  "val_start_precision": 0.7856,
  "val_start_recall": 0.6912,
  "val_start_f1": 0.7354,
  "val_end_accuracy": 0.9187,
  "val_end_precision": 0.7723,
  "val_end_recall": 0.6834,
  "val_end_f1": 0.7249,
  "val_span_precision": 0.6234,
  "val_span_recall": 0.5891,
  "val_span_f1": 0.6057,
  "val_exact_matches": 123,
  "val_partial_matches": 45,
  "val_total_predictions": 197,
  "val_total_ground_truth": 209,
  "test_start_accuracy": 0.9156,
  "test_start_precision": 0.7634,
  "test_start_recall": 0.6745,
  "test_start_f1": 0.7158,
  ...
}
```

## Metrics Explained

### Token-Level Metrics

These metrics evaluate the model's ability to predict start and end token positions:

- **start_accuracy**: Accuracy for predicting start token positions
- **start_precision**: Precision for start token predictions
- **start_recall**: Recall for start token predictions
- **start_f1**: F1 score for start token predictions
- **end_accuracy/precision/recall/f1**: Same as above but for end tokens

### Span-Level Metrics

These metrics evaluate complete evidence span extraction:

- **span_precision**: Proportion of predicted spans that exactly match ground truth
- **span_recall**: Proportion of ground truth spans that were correctly predicted
- **span_f1**: Harmonic mean of span precision and recall
- **exact_matches**: Number of predicted spans that exactly match ground truth spans
- **partial_matches**: Number of predicted spans with partial overlap (but not exact match)
- **total_predictions**: Total number of spans predicted by the model
- **total_ground_truth**: Total number of ground truth spans in the dataset

## Interpreting Results

### Good Performance Indicators

- **Span F1 > 0.6**: Model is reasonably good at identifying evidence spans
- **Start/End F1 > 0.7**: Model accurately predicts token boundaries
- **High exact_matches**: Model predictions align well with annotations

### Common Issues

1. **Low span_recall, high span_precision**: Model is too conservative (predicts too few spans)
   - Solution: Lower `--span_threshold`

2. **High span_recall, low span_precision**: Model is too aggressive (predicts too many spans)
   - Solution: Raise `--span_threshold`

3. **Many partial_matches, few exact_matches**: Model identifies evidence regions but boundaries are off
   - Solution: May need more training or better boundary annotations

## Example Usage Workflow

```bash
# 1. Run evaluation with default settings
make evaluate-evidence-detailed

# 2. Check the results
cat outputs/train_Evidence/evaluation/evaluation_metrics.json

# 3. View predictions in Excel or pandas
python -c "import pandas as pd; df = pd.read_csv('outputs/train_Evidence/evaluation/val_predictions.csv'); print(df.head())"

# 4. If threshold needs adjustment, re-run with custom threshold
PYTHONPATH=/experiment/YuNing/DataAugmentation_Evaluation python scripts/evaluate_evidence_detailed.py \
    --checkpoint_dir outputs/train_Evidence \
    --output_dir outputs/train_Evidence/evaluation_threshold_0.6 \
    --span_threshold 0.6
```

## Analyzing Predictions

### Load and analyze predictions in Python:

```python
import pandas as pd
import json

# Load predictions
val_df = pd.read_csv('outputs/train_Evidence/evaluation/val_predictions.csv')
test_df = pd.read_csv('outputs/train_Evidence/evaluation/test_predictions.csv')

# Load metrics
with open('outputs/train_Evidence/evaluation/evaluation_metrics.json') as f:
    metrics = json.load(f)

# Examples with correct predictions
correct = val_df[val_df['num_predicted_spans'] == val_df['num_ground_truth_spans']]
print(f"Examples with correct span count: {len(correct)}")

# Examples where model missed evidence
false_negatives = val_df[(val_df['has_evidence'] == True) & (val_df['num_predicted_spans'] == 0)]
print(f"False negatives: {len(false_negatives)}")

# Examples where model hallucinated evidence
false_positives = val_df[(val_df['has_evidence'] == False) & (val_df['num_predicted_spans'] > 0)]
print(f"False positives: {len(false_positives)}")

# View specific examples
print("\nExample predictions:")
for idx, row in val_df.head(3).iterrows():
    print(f"\nPost ID: {row['post_id']}")
    print(f"Criterion: {row['criterion']}")
    print(f"Predicted: {row['predicted_sentences']}")
    print(f"Ground Truth: {row['ground_truth_sentences']}")
```

## Troubleshooting

### Error: "Checkpoint not found"
- Check that `outputs/train_Evidence/best/model.pt` exists
- Verify the `--checkpoint_dir` path is correct

### Error: "ModuleNotFoundError: No module named 'src'"
- Make sure to set `PYTHONPATH` or use `make evaluate-evidence-detailed`

### Low memory or GPU OOM
- Reduce `--batch_size` (try 8 or 4)

### Unexpected results
- Verify the model was trained properly
- Check that data splits match training (seed=42)
- Ensure `--model_name` matches the model used during training

## Integration with MLflow

While detailed predictions are saved locally, the evaluation can also be logged to MLflow:

```python
import mlflow
from scripts.evaluate_evidence_detailed import main

with mlflow.start_run(run_name="evidence_evaluation"):
    # Run evaluation
    main()

    # Log CSV files as artifacts
    mlflow.log_artifact("outputs/train_Evidence/evaluation/val_predictions.csv")
    mlflow.log_artifact("outputs/train_Evidence/evaluation/test_predictions.csv")
    mlflow.log_artifact("outputs/train_Evidence/evaluation/evaluation_metrics.json")
```

## Next Steps

After evaluation:

1. **Analyze errors**: Use the CSV files to identify failure modes
2. **Adjust threshold**: Experiment with different `--span_threshold` values
3. **Improve training**: Use insights to refine training data or model architecture
4. **Deploy model**: If metrics are satisfactory, proceed to production deployment
