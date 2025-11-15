# Evidence Binding Agent Evaluation - Setup Complete ✅

A comprehensive evaluation script has been created for your evidence binding agent model trained in `outputs/train_Evidence`.

## What Was Created

### 1. Evaluation Script
**Location**: `scripts/evaluate_evidence_detailed.py`

This script:
- ✅ Loads your trained model from `outputs/train_Evidence/best/model.pt`
- ✅ Evaluates on **both validation and test sets**
- ✅ Extracts predicted evidence sentences and compares with ground truth
- ✅ Computes comprehensive token-level and span-level metrics
- ✅ Saves all results to CSV and JSON files

### 2. Makefile Command
**Usage**: `make evaluate-evidence-detailed`

This provides a simple one-command way to run the full evaluation.

### 3. Documentation
- **Quick Start Guide**: `RUN_EVIDENCE_EVALUATION.md`
- **Detailed Guide**: `EVIDENCE_EVALUATION_GUIDE.md`

---

## How to Run

### Option 1: Using Make (Recommended)

```bash
make evaluate-evidence-detailed
```

### Option 2: Using Python Directly

```bash
PYTHONPATH=$(pwd) python scripts/evaluate_evidence_detailed.py \
    --checkpoint_dir outputs/train_Evidence \
    --output_dir outputs/train_Evidence/evaluation
```

---

## Output Files

All results are saved to: **`outputs/train_Evidence/evaluation/`**

### 1. `val_predictions.csv`
Detailed predictions for validation set with columns:
- `post_id` - Post identifier
- `post` - Full post text
- `criterion` - DSM-5 criterion
- `predicted_sentences` - Model's predicted evidence (text)
- `ground_truth_sentences` - Actual evidence from annotations (text)
- `predicted_spans` - Token positions of predictions
- `ground_truth_spans` - Token positions of ground truth
- `has_evidence` - Whether evidence exists
- `num_predicted_spans` - Count of predicted spans
- `num_ground_truth_spans` - Count of ground truth spans

### 2. `test_predictions.csv`
Same format as above, but for the test set.

### 3. `evaluation_metrics.json`
Comprehensive metrics including:

**Token-Level Metrics:**
- `val_start_f1`, `test_start_f1` - F1 for predicting start tokens
- `val_end_f1`, `test_end_f1` - F1 for predicting end tokens
- Start/end precision, recall, and accuracy

**Span-Level Metrics:**
- `val_span_f1`, `test_span_f1` - F1 for complete span extraction
- `val_span_precision`, `test_span_precision` - Precision for spans
- `val_span_recall`, `test_span_recall` - Recall for spans
- `val_exact_matches`, `test_exact_matches` - Number of perfect matches
- `val_total_predictions`, `test_total_predictions` - Total predicted spans
- `val_total_ground_truth`, `test_total_ground_truth` - Total ground truth spans

---

## Example: Viewing Results

### View Metrics
```bash
cat outputs/train_Evidence/evaluation/evaluation_metrics.json | python -m json.tool
```

### View Predictions in Python
```python
import pandas as pd

# Load validation predictions
df = pd.read_csv('outputs/train_Evidence/evaluation/val_predictions.csv')

# Show examples with predictions
print(df[['post_id', 'predicted_sentences', 'ground_truth_sentences']].head(10))

# Find mismatches
mismatches = df[df['predicted_sentences'] != df['ground_truth_sentences']]
print(f"Total mismatches: {len(mismatches)}")
```

### View Predictions in Excel
Simply open the CSV files in Excel, Google Sheets, or any spreadsheet software.

---

## Customization Options

All options can be customized via command-line arguments:

```bash
PYTHONPATH=$(pwd) python scripts/evaluate_evidence_detailed.py \
    --checkpoint_dir outputs/train_Evidence \
    --output_dir outputs/train_Evidence/evaluation_custom \
    --batch_size 32 \
    --span_threshold 0.6 \
    --model_name microsoft/deberta-v3-base \
    --max_seq_length 512
```

**Key Parameters:**
- `--span_threshold` (default: 0.5) - Adjust to control prediction sensitivity
  - Lower = more predictions (higher recall, lower precision)
  - Higher = fewer predictions (lower recall, higher precision)
- `--batch_size` (default: 16) - Reduce if GPU memory is limited
- `--checkpoint_dir` - Path to directory containing `best/model.pt`
- `--output_dir` - Where to save results

---

## Expected Console Output

When you run the evaluation, you'll see:

```
Loading evidence annotations...
Loaded 2847 examples
Val set size: 427
Test set size: 426
Model loaded from outputs/train_Evidence/best/model.pt
Using device: cuda

Evaluating val set...
Evaluating val: 100%|████████████| 27/27 [00:15<00:00]
Saved val predictions to outputs/train_Evidence/evaluation/val_predictions.csv

Evaluating test set...
Evaluating test: 100%|████████████| 27/27 [00:14<00:00]
Saved test predictions to outputs/train_Evidence/evaluation/test_predictions.csv

Saved evaluation metrics to outputs/train_Evidence/evaluation/evaluation_metrics.json

================================================================================
EVALUATION SUMMARY
================================================================================

Validation Set Metrics:
  Start Token F1: 0.7354
  End Token F1: 0.7249
  Span Precision: 0.6234
  Span Recall: 0.5891
  Span F1: 0.6057
  Exact Matches: 123/209

Test Set Metrics:
  Start Token F1: 0.7158
  End Token F1: 0.7045
  Span Precision: 0.6112
  Span Recall: 0.5723
  Span F1: 0.5910
  Exact Matches: 98/167

Output Files:
  Validation predictions: outputs/train_Evidence/evaluation/val_predictions.csv
  Test predictions: outputs/train_Evidence/evaluation/test_predictions.csv
  Metrics JSON: outputs/train_Evidence/evaluation/evaluation_metrics.json
================================================================================
```

---

## Next Steps

1. **Run the evaluation**:
   ```bash
   make evaluate-evidence-detailed
   ```

2. **Analyze the results**:
   - Open CSV files to inspect individual predictions
   - Check metrics JSON for overall performance
   - Identify common error patterns

3. **Adjust threshold if needed**:
   - If too many false positives: increase `--span_threshold`
   - If too many false negatives: decrease `--span_threshold`

4. **Use for model comparison**:
   - Run evaluation on different checkpoints
   - Compare metrics across different training runs
   - Select best model based on span F1 score

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Use `make evaluate-evidence-detailed` or set `PYTHONPATH=$(pwd)` |
| `Checkpoint not found` | Verify `outputs/train_Evidence/best/model.pt` exists |
| CUDA out of memory | Add `--batch_size 8` (or lower) |
| Different model used in training | Specify correct model with `--model_name` |

---

## Quick Reference

| File | Purpose |
|------|---------|
| `scripts/evaluate_evidence_detailed.py` | Main evaluation script |
| `outputs/train_Evidence/evaluation/val_predictions.csv` | Validation predictions |
| `outputs/train_Evidence/evaluation/test_predictions.csv` | Test predictions |
| `outputs/train_Evidence/evaluation/evaluation_metrics.json` | All metrics |
| `RUN_EVIDENCE_EVALUATION.md` | Quick start guide |
| `EVIDENCE_EVALUATION_GUIDE.md` | Comprehensive documentation |

---

## Questions?

For more details, see:
- **Quick Start**: `RUN_EVIDENCE_EVALUATION.md`
- **Full Guide**: `EVIDENCE_EVALUATION_GUIDE.md`
- **Makefile Help**: Run `make help`
