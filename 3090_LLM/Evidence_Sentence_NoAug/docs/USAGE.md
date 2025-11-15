# Usage Guide

Comprehensive guide for using the DeBERTa-v3 Evidence Sentence Classification pipeline.

## Table of Contents

1. [Installation](#installation)
2. [Data Format](#data-format)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Inference](#inference)
6. [Configuration](#configuration)
7. [MLflow Integration](#mlflow-integration)
8. [Advanced Usage](#advanced-usage)

## Installation

### Standard Installation

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install --upgrade pip
pip install -e .
```

### With Development Dependencies

```bash
pip install -e '.[dev]'
```

This includes: `pytest`, `ruff`, `black`, `mypy` for testing and code quality.

## Data Format

### Required Data Files

The pipeline expects three data sources:

#### 1. DSM-5 Criteria (JSON)

Location: `data/DSM5/*.json`

Format:
```json
{
  "criterion_id": "MDD_A1",
  "text": "Depressed mood most of the day, nearly every day",
  "category": "Major Depressive Disorder"
}
```

#### 2. Posts CSV

Location: `data/redsm5/posts.csv`

Columns:
- `post_id`: Unique post identifier
- `text`: Full post text

#### 3. Annotations CSV

Location: `data/redsm5/annotations.csv`

Columns:
- `post_id`: Links to posts.csv
- `sentence_id`: Unique sentence identifier
- `sentence_text`: The annotated sentence
- `DSM5_symptom`: Criterion identifier
- `status`: 1 (evidence) or 0 (not evidence)
- `explanation`: Clinical rationale (optional)

### Data Validation

Test data loading before training:

```bash
python scripts/test_data_loading.py
```

This script:
- Loads and validates all data files
- Applies negative sampling
- Creates cross-validation folds
- Reports data statistics

## Training

### Basic Training

Run with default configuration:

```bash
python scripts/train.py
```

Default settings:
- Model: `microsoft/deberta-v3-base`
- Epochs: 3
- Batch size: 16 (train), 32 (eval)
- Learning rate: 2e-5
- Loss: Weighted cross-entropy
- CV folds: 5
- Seed: 1337

### Custom Hyperparameters

Override any configuration parameter:

```bash
python scripts/train.py \
  training.args.num_train_epochs=5 \
  training.args.learning_rate=3e-5 \
  training.args.per_device_train_batch_size=8 \
  training.args.weight_decay=0.01 \
  training.args.warmup_ratio=0.1
```

### Loss Functions

#### Weighted Cross-Entropy (Default)

```bash
python scripts/train.py loss=weighted_ce
```

Automatically computes class weights from training data to handle imbalance.

#### Focal Loss

```bash
python scripts/train.py loss=focal
```

Uses focal loss with gamma=2.0. Good for highly imbalanced datasets.

Adjust gamma:
```bash
python scripts/train.py loss=focal loss.gamma=3.0
```

### Data Configuration

#### Adjust Negative Sampling Ratio

Default is 1:3 (positive:negative):

```bash
python scripts/train.py data.pos_neg_ratio=2  # 1:2 ratio
```

#### Change Maximum Token Length

Default is 512 tokens:

```bash
python scripts/train.py data.max_length=256
```

### Cross-Validation Settings

#### Change Number of Folds

```bash
python scripts/train.py training.n_folds=10
```

#### Change Random Seed

```bash
python scripts/train.py training.seed=42
```

### Training Output

The training process creates:

1. **Model Checkpoints**: `outputs/models/fold_0/` through `outputs/models/fold_4/`
2. **MLflow Runs**: Parent run with 5 child runs
3. **Logs**: Console output with training progress
4. **Results**: `cv_results.json` with aggregate metrics

Example console output:

```
Fold 1/5
Training: 100%|██████████| 304/304 [05:23<00:00]
Validation: 100%|██████████| 76/76 [00:32<00:00]
eval_loss: 0.3421, eval_f1_macro: 0.8234, eval_accuracy: 0.8512

...

Best Fold: 2
Best Model Path: outputs/models/fold_2
Aggregate f1_macro: 0.8312 ± 0.0145
```

## Evaluation

### Metrics Computed

The pipeline automatically computes:

**Classification Metrics**:
- Macro-F1 (primary metric for model selection)
- Accuracy
- Per-class Precision, Recall, F1

**Probabilistic Metrics**:
- ROC-AUC
- PR-AUC

**Confusion Matrix**:
- True Positives, True Negatives
- False Positives, False Negatives

### Viewing Results

#### Console Output

Training script prints aggregate results:

```
CROSS-VALIDATION RESULTS
Best Fold: 2
Best Model Path: outputs/models/fold_2

Best Fold Metrics:
  eval_f1_macro: 0.8421
  eval_accuracy: 0.8567
  eval_roc_auc: 0.9123

Aggregate Metrics:
  f1_macro: 0.8312 ± 0.0145
  accuracy: 0.8489 ± 0.0098
  roc_auc: 0.9056 ± 0.0087
```

#### MLflow UI

1. Start MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db`
2. Navigate to http://127.0.0.1:5000
3. Click on the `evidence-binding-cv` experiment
4. View parent and child runs with detailed metrics

#### Result Files

`cv_results.json` contains structured results:

```json
{
  "best_fold_index": 2,
  "best_fold_metrics": {
    "eval_f1_macro": 0.8421,
    "eval_accuracy": 0.8567,
    "eval_roc_auc": 0.9123
  },
  "aggregate_metrics": {
    "f1_macro": {"mean": 0.8312, "std": 0.0145},
    "accuracy": {"mean": 0.8489, "std": 0.0098},
    "roc_auc": {"mean": 0.9056, "std": 0.0087}
  },
  "fold_metrics": [...]
}
```

## Inference

### Single Prediction

Predict on a single criterion-sentence pair:

```bash
python scripts/inference.py \
  --criterion "Depressed mood most of the day, nearly every day" \
  --sentence "I've been feeling really down lately" \
  --model-path outputs/models/fold_2/best_model
```

Output:
```
Criterion: Depressed mood most of the day, nearly every day
Sentence: I've been feeling really down lately
Prediction: 1 (evidence)
Probability: 0.8734
```

### Batch Inference

Create an input JSON file (`input_pairs.json`):

```json
[
  {
    "criterion": "Depressed mood most of the day",
    "sentence": "I feel sad all the time"
  },
  {
    "criterion": "Markedly diminished interest or pleasure",
    "sentence": "I don't enjoy anything anymore"
  },
  {
    "criterion": "Significant weight loss or gain",
    "sentence": "I've lost about 20 pounds recently"
  }
]
```

Run batch inference:

```bash
python scripts/inference.py \
  --batch-file input_pairs.json \
  --model-path outputs/models/fold_2/best_model \
  --output predictions.json
```

Output file (`predictions.json`):

```json
[
  {
    "criterion": "Depressed mood most of the day",
    "sentence": "I feel sad all the time",
    "prediction": 1,
    "probability": 0.9234
  },
  {
    "criterion": "Markedly diminished interest or pleasure",
    "sentence": "I don't enjoy anything anymore",
    "prediction": 1,
    "probability": 0.8876
  },
  {
    "criterion": "Significant weight loss or gain",
    "sentence": "I've lost about 20 pounds recently",
    "prediction": 1,
    "probability": 0.7654
  }
]
```

### Programmatic Inference

Use the trained model in Python code:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
model_path = "outputs/models/fold_2/best_model"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Prepare input
criterion = "Depressed mood most of the day"
sentence = "I feel sad all the time"
text = f"{criterion} [SEP] {sentence}"

# Tokenize
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Predict
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    prediction = torch.argmax(probs, dim=-1).item()
    probability = probs[0, 1].item()

print(f"Prediction: {prediction}, Probability: {probability:.4f}")
```

## Configuration

### Configuration Structure

The project uses Hydra for configuration management with modular config files:

```
configs/
├── config.yaml           # Main config with defaults
├── model/
│   └── deberta_v3.yaml  # Model configuration
├── trainer/
│   └── cv.yaml          # Training arguments
├── loss/
│   ├── weighted_ce.yaml # Weighted CE config
│   └── focal.yaml       # Focal loss config
├── cv/
│   └── default.yaml     # CV settings
├── data/
│   └── evidence_pairs.yaml  # Data config
└── logger/
    └── mlflow.yaml      # MLflow config
```

### Overriding Configurations

#### Via CLI

```bash
python scripts/train.py \
  model.name=microsoft/deberta-v3-large \
  training.args.learning_rate=1e-5 \
  data.max_length=256
```

#### Via Config Files

Create a custom config file `configs/my_experiment.yaml`:

```yaml
defaults:
  - config
  - override model: deberta_v3
  - override loss: focal

training:
  args:
    num_train_epochs: 5
    learning_rate: 1e-5
    per_device_train_batch_size: 8

data:
  pos_neg_ratio: 2
```

Use it:
```bash
python scripts/train.py --config-name my_experiment
```

### Key Configuration Parameters

**Model** (`configs/model/deberta_v3.yaml`):
```yaml
model:
  name: "microsoft/deberta-v3-base"
  num_labels: 2
  problem_type: "single_label_classification"
```

**Training** (`configs/trainer/cv.yaml`):
```yaml
training:
  seed: 1337
  n_folds: 5
  args:
    num_train_epochs: 3
    learning_rate: 2.0e-5
    per_device_train_batch_size: 16
    warmup_ratio: 0.06
    weight_decay: 0.01
```

**Data** (`configs/data/evidence_pairs.yaml`):
```yaml
data:
  dsm5_dir: "data/DSM5"
  csv_path: "data/redsm5/posts.csv"
  pos_neg_ratio: 3
  max_length: 512
```

## MLflow Integration

### Starting MLflow UI

```bash
mlflow ui \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns
```

Access at: http://127.0.0.1:5000

### What Gets Logged

**Parameters**:
- Model name, learning rate, batch size
- Number of folds, seed
- Loss type and parameters
- Git SHA, CUDA version

**Metrics** (per epoch):
- Training loss
- Validation metrics (F1, accuracy, ROC-AUC, etc.)

**Artifacts**:
- Model checkpoints
- Tokenizer files
- Configuration YAML
- Requirements.txt (pip freeze)
- cv_results.json

**Tags**:
- Experiment name
- Fold number
- Best fold indicator

### Run Structure

```
Parent Run: "cross_validation"
├── Child Run: "fold_0"
│   ├── Metrics: per-epoch training/validation
│   └── Artifacts: fold_0 model checkpoint
├── Child Run: "fold_1"
├── Child Run: "fold_2"
├── Child Run: "fold_3"
└── Child Run: "fold_4"
```

### Querying Runs

Using MLflow Python API:

```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Get experiment
experiment = mlflow.get_experiment_by_name("evidence-binding-cv")

# Search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.eval_f1_macro DESC"]
)

print(runs[['run_id', 'metrics.eval_f1_macro', 'params.learning_rate']])
```

## Advanced Usage

### Custom Data Split

The pipeline uses GroupKFold by `post_id` to prevent data leakage. To customize:

Edit `src/Project/SubProject/data/dataset.py`:

```python
def create_folds(df, n_splits=5, seed=42, group_col='post_id'):
    # Custom split logic here
    pass
```

### Adding Custom Metrics

Edit `src/Project/SubProject/engine/eval_engine.py`:

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Add custom metrics
    from sklearn.metrics import matthews_corrcoef

    metrics = {
        # ... existing metrics ...
        'matthews_corrcoef': matthews_corrcoef(labels, predictions)
    }
    return metrics
```

### Early Stopping

Configure in `configs/trainer/cv.yaml`:

```yaml
training:
  args:
    load_best_model_at_end: true
    metric_for_best_model: "eval_f1_macro"
    greater_is_better: true
    save_strategy: "epoch"
    evaluation_strategy: "epoch"
    early_stopping_patience: 3  # Stop after 3 epochs without improvement
```

### Gradient Accumulation

For larger effective batch sizes with limited GPU memory:

```bash
python scripts/train.py \
  training.args.per_device_train_batch_size=4 \
  training.args.gradient_accumulation_steps=4
  # Effective batch size = 4 * 4 = 16
```

### Mixed Precision Training

Automatically handled by the training script. It tries:
1. BF16 (best for A100, H100)
2. FP16 (good for V100, RTX series)
3. FP32 (fallback)

### Learning Rate Scheduling

The pipeline uses linear decay with warmup. Customize:

```bash
python scripts/train.py \
  training.args.lr_scheduler_type=cosine \
  training.args.warmup_ratio=0.1
```

Available schedulers: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, `constant_with_warmup`

### Optimizer Selection

The training script auto-detects the best available optimizer. To force a specific one, modify the training args in config files.

## Troubleshooting

### Common Issues

**Out of Memory**:
```bash
# Reduce batch size
python scripts/train.py training.args.per_device_train_batch_size=4

# Or use gradient accumulation
python scripts/train.py \
  training.args.per_device_train_batch_size=4 \
  training.args.gradient_accumulation_steps=4
```

**Slow Training**:
- Check if GPU is being used
- Reduce `data.max_length` if sequences are too long
- Increase `training.args.dataloader_num_workers`

**Poor Performance**:
- Increase epochs: `training.args.num_train_epochs=5`
- Adjust learning rate: try 1e-5, 3e-5, 5e-5
- Change loss function: try `loss=focal`
- Adjust negative sampling: `data.pos_neg_ratio=2`

**Data Loading Errors**:
- Verify file paths in `configs/data/evidence_pairs.yaml`
- Run `python scripts/test_data_loading.py` for diagnostics
- Check CSV file encoding (should be UTF-8)

## Best Practices

1. **Always test data loading first**: `python scripts/test_data_loading.py`
2. **Start MLflow UI before training** to monitor progress in real-time
3. **Use version control** for config files to track experiments
4. **Set seeds** for reproducibility: `training.seed=1337`
5. **Monitor validation metrics** during training to detect overfitting
6. **Save experiment configs** via MLflow for later reproduction
7. **Test inference** on a few examples before batch processing

## Next Steps

- Explore hyperparameter tuning with different learning rates and batch sizes
- Try different model architectures (deberta-v3-large, roberta-base)
- Analyze misclassifications to understand model limitations
- Export best model for production deployment
- Add custom evaluation metrics for domain-specific needs
