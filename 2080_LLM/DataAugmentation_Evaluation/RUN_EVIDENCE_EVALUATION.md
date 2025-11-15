# Quick Start: Evidence Binding Agent Evaluation

## Run Evaluation (Simplest Method)

```bash
make evaluate-evidence-detailed
```

This single command will:
- ✅ Load trained model from `outputs/train_Evidence/best/model.pt`
- ✅ Evaluate on **validation set**
- ✅ Evaluate on **test set**
- ✅ Save detailed predictions to CSV files
- ✅ Save comprehensive metrics to JSON
- ✅ Print summary to console

## Output Location

All results are saved to: `outputs/train_Evidence/evaluation/`

### Files Generated:

1. **`val_predictions.csv`** - Detailed validation set predictions
   - Columns: post_id, post, criterion, predicted_sentences, ground_truth_sentences, etc.

2. **`test_predictions.csv`** - Detailed test set predictions
   - Same format as validation predictions

3. **`evaluation_metrics.json`** - Complete metrics for both sets
   - Token-level metrics (start/end F1, precision, recall)
   - Span-level metrics (span F1, exact matches, etc.)

## Alternative: Custom Settings

```bash
# With custom threshold
PYTHONPATH=$(pwd) python scripts/evaluate_evidence_detailed.py \
    --checkpoint_dir outputs/train_Evidence \
    --output_dir outputs/train_Evidence/evaluation \
    --span_threshold 0.6 \
    --batch_size 32

# Different checkpoint location
PYTHONPATH=$(pwd) python scripts/evaluate_evidence_detailed.py \
    --checkpoint_dir outputs/evidence/20251014_102030 \
    --output_dir outputs/evidence/20251014_102030/evaluation
```

## View Results

```bash
# View metrics
cat outputs/train_Evidence/evaluation/evaluation_metrics.json

# View predictions (first 10 rows)
head -10 outputs/train_Evidence/evaluation/val_predictions.csv

# Load in Python
python -c "
import pandas as pd
df = pd.read_csv('outputs/train_Evidence/evaluation/val_predictions.csv')
print(df[['post_id', 'predicted_sentences', 'ground_truth_sentences']].head())
"
```

## Expected Output

The script will print a summary like:

```
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

## Troubleshooting

**Issue**: Command not found or module errors
- **Solution**: Make sure you're in the project root directory

**Issue**: CUDA out of memory
- **Solution**: Add `--batch_size 8` (or lower)

**Issue**: Model checkpoint not found
- **Solution**: Verify `outputs/train_Evidence/best/model.pt` exists

## Full Documentation

For detailed information about metrics, output format, and analysis examples, see:
**[EVIDENCE_EVALUATION_GUIDE.md](./EVIDENCE_EVALUATION_GUIDE.md)**
