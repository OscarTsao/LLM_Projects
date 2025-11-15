# Quickstart — 5-Fold DeBERTaV3 Evidence Binding

This guide walks through the complete workflow from installation to inference using the implemented DeBERTa-v3 evidence classification pipeline.

## 0. Prerequisites

1. **Python 3.10+** with virtualenv
2. **Install dependencies**:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install --upgrade pip
   pip install -e .
   ```

   Required packages: `torch`, `transformers>=4.40`, `hydra-core`, `omegaconf`, `mlflow>=2.8`, `pandas`, `scikit-learn`

3. **Verify data files**:
   - Criteria JSON: `data/DSM5/MDD_Criteria.json`
   - Posts: `data/redsm5/posts.csv`
   - Annotations: `data/redsm5/annotations.csv`

4. **Launch MLflow UI** (optional but recommended) in a separate terminal:
   ```bash
   mlflow ui \
     --backend-store-uri sqlite:///mlflow.db \
     --default-artifact-root ./mlruns
   ```
   Access at: http://127.0.0.1:5000

## 1. Test Data Loading

Verify the data pipeline before training:

```bash
python scripts/test_data_loading.py
```

**Expected output**:
```
Loading DSM-5 criteria from data/DSM5...
Loaded 9 DSM-5 criteria
Loading ReDSM5 data from data/redsm5/posts.csv...
Loaded 1484 posts with 1547 annotations
Applying stratified negative sampling (1:3 ratio)...
Created 5 folds with GroupKFold
Fold 0: train=4856, val=1214
Fold 1: train=4856, val=1214
...
Data loading test completed successfully!
```

## 2. Train 5-Fold Cross-Validation

### Basic Training (Default Config)

```bash
python scripts/train.py
```

Uses defaults from `configs/config.yaml`:
- Model: `microsoft/deberta-v3-base`
- Loss: Weighted cross-entropy
- Epochs: 3
- Learning rate: 2e-5
- Batch size: 16 (train), 32 (eval)
- Warmup: 0.06
- Folds: 5
- Seed: 1337

### Custom Configuration

Override any parameter via Hydra CLI:

```bash
python scripts/train.py \
  training.args.num_train_epochs=5 \
  training.args.learning_rate=3e-5 \
  training.args.per_device_train_batch_size=8 \
  data.pos_neg_ratio=2
```

### Use Focal Loss

Switch to focal loss for handling class imbalance:

```bash
python scripts/train.py loss=focal
```

### Training Outputs

The training script produces:

1. **MLflow Runs**:
   - Parent run: Cross-validation orchestration
   - 5 child runs: Individual fold training

2. **Model Checkpoints**:
   - `outputs/models/fold_0/` through `outputs/models/fold_4/`
   - Each contains `best_model/` directory with model, tokenizer, and config

3. **Logged Artifacts** (in MLflow):
   - `config.yaml`: Complete Hydra configuration
   - `requirements.txt`: Pip freeze output
   - `cv_results.json`: Aggregate metrics and best fold info
   - Environment metadata (git SHA, CUDA version, etc.)

4. **Console Output**:
   ```
   ================================================================================
   CROSS-VALIDATION RESULTS
   ================================================================================
   Best Fold: 2
   Best Model Path: outputs/models/fold_2

   Best Fold Metrics:
     eval_loss: 0.3245
     eval_f1_macro: 0.8421
     eval_accuracy: 0.8567
     eval_roc_auc: 0.9123

   Aggregate Metrics:
     f1_macro: 0.8312 ± 0.0145
     accuracy: 0.8489 ± 0.0098
     roc_auc: 0.9056 ± 0.0087
   ================================================================================
   ```

## 3. Review Results in MLflow

1. Open MLflow UI at http://127.0.0.1:5000
2. Navigate to the `evidence-binding-cv` experiment
3. Inspect:
   - Parent run: Aggregate metrics, best fold selection
   - Child runs: Per-fold training curves and metrics
   - Artifacts: Models, configs, and result files

**Key Metrics Logged**:
- Macro-F1 (primary metric for best fold selection)
- Accuracy
- ROC-AUC
- PR-AUC
- Positive class F1, Precision, Recall
- Confusion matrix values

## 4. Run Inference

### Single Prediction

```bash
python scripts/inference.py \
  --criterion "Depressed mood most of the day, nearly every day" \
  --sentence "I feel sad and empty all the time" \
  --model-path outputs/models/fold_2/best_model
```

**Expected output**:
```
Criterion: Depressed mood most of the day, nearly every day
Sentence: I feel sad and empty all the time
Prediction: 1 (evidence)
Probability: 0.9234
```

### Batch Inference

Create a JSON file with multiple pairs:

```json
[
  {
    "criterion": "Depressed mood most of the day",
    "sentence": "I feel sad all the time"
  },
  {
    "criterion": "Markedly diminished interest or pleasure",
    "sentence": "I don't enjoy anything anymore"
  }
]
```

Run batch inference:

```bash
python scripts/inference.py \
  --batch-file inputs.json \
  --model-path outputs/models/fold_2/best_model \
  --output results.json
```

## 5. Configuration Details

### Available Config Groups

**Model** (`configs/model/`):
- `deberta_v3.yaml`: microsoft/deberta-v3-base (default)

**Loss** (`configs/loss/`):
- `weighted_ce.yaml`: Weighted cross-entropy (default)
- `focal.yaml`: Focal loss with gamma=2.0

**Trainer** (`configs/trainer/`):
- `cv.yaml`: Cross-validation training arguments

**Data** (`configs/data/`):
- `evidence_pairs.yaml`: Data loading and sampling config

### Key Parameters

**Training**:
- `training.seed`: Random seed (default: 1337)
- `training.n_folds`: Number of CV folds (default: 5)
- `training.args.num_train_epochs`: Epochs per fold (default: 3)
- `training.args.learning_rate`: Learning rate (default: 2e-5)

**Data**:
- `data.pos_neg_ratio`: Positive to negative sampling ratio (default: 3)
- `data.max_length`: Maximum token length (default: 512)

**Loss**:
- `loss.type`: 'weighted_ce' or 'focal'
- `loss.gamma`: Focal loss gamma parameter (default: 2.0)

## 6. Troubleshooting

### Out of Memory
- Reduce batch size: `training.args.per_device_train_batch_size=8`
- Enable gradient accumulation: `training.args.gradient_accumulation_steps=2`

### CPU-Only Training
- PyTorch auto-detects CPU/GPU
- Training will be slower but functional

### Precision Fallback
The training script automatically detects supported precision:
1. Tries BF16 (if supported)
2. Falls back to FP16 (if supported)
3. Falls back to FP32

### Missing Dependencies
```bash
pip install torch transformers hydra-core omegaconf mlflow pandas scikit-learn
```

### Data File Issues
Verify file paths in `configs/data/evidence_pairs.yaml`:
- `dsm5_dir: "data/DSM5"`
- `csv_path: "data/redsm5/posts.csv"`

## 7. Next Steps

1. **Hyperparameter tuning**: Experiment with learning rates, batch sizes, epochs
2. **Model selection**: Try other DeBERTa variants or BERT models
3. **Error analysis**: Inspect misclassifications in validation folds
4. **Production deployment**: Export best model for serving
5. **Additional metrics**: Add custom metrics in `eval_engine.py`

## 8. Validation Checklist

Before considering the implementation complete:

- ✅ Data loading test passes
- ✅ Training completes all 5 folds successfully
- ✅ MLflow UI shows parent + child runs with metrics
- ✅ Best fold identified and metrics aggregated
- ✅ Inference produces reasonable predictions
- ⚠️ Unit tests (to be implemented)
- ⚠️ Extended documentation (in progress)
