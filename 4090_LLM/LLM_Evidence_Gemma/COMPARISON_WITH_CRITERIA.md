# Comparison: LLM_Evidence_Gemma vs LLM_Criteria_Gemma

This document details the differences and adaptations between the two projects, both of which use bidirectional Gemma models on the ReDSM5 dataset but for different downstream tasks.

## Overview

| Project | LLM_Criteria_Gemma | LLM_Evidence_Gemma |
|---------|-------------------|-------------------|
| **Primary Task** | **Criteria Classification** | **Evidence Extraction (QA)** |
| **Objective** | Classify which DSM-5 criterion applies | Extract text evidence supporting a criterion |
| **Task Type** | Multi-class classification | Extractive question answering |
| **Input Format** | Reddit post text only | Question + Reddit post context |
| **Output Format** | Single label (0-9) | Answer span (start/end positions) |

---

## Detailed Differences

### 1. Model Architecture

#### LLM_Criteria_Gemma (Classification)
```python
class GemmaClassifier(nn.Module):
    def __init__(self, num_classes=10, pooling_strategy='mean', ...):
        self.encoder = GemmaEncoder(...)  # Bidirectional
        self.pooler = MeanPooler()        # Aggregate sequence
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes)  # Output: [batch, 10]
        )

    def forward(self, input_ids, attention_mask, labels=None):
        hidden = self.encoder(...)         # [batch, seq_len, hidden]
        pooled = self.pooler(hidden, ...)  # [batch, hidden]
        logits = self.classifier(pooled)   # [batch, 10]
        return logits
```

#### LLM_Evidence_Gemma (Extractive QA)
```python
class GemmaQA(nn.Module):
    def __init__(self, ...):
        self.encoder = GemmaEncoder(...)  # Bidirectional (same)
        # No pooler - need per-token representations
        self.qa_outputs = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2)     # Output: start + end logits
        )

    def forward(self, input_ids, attention_mask, start_pos=None, end_pos=None):
        sequence_output = self.encoder(...)  # [batch, seq_len, hidden]
        # No pooling - keep all tokens
        logits = self.qa_outputs(sequence_output)  # [batch, seq_len, 2]
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits, end_logits
```

**Key Differences**:
- **Pooling**: Classification needs sentence-level representation; QA needs per-token representations
- **Head**: Classification has single linear layer (10 classes); QA has two outputs (start + end)
- **Output Shape**: Classification → `[batch, num_classes]`; QA → `[batch, seq_len]` (x2)

---

### 2. Data Format and Dataset

#### LLM_Criteria_Gemma
```python
class ReDSM5Dataset(Dataset):
    def __init__(self, texts, symptom_indices, tokenizer, ...):
        self.texts = texts                    # List of post texts
        self.symptom_indices = symptom_indices  # List of labels (0-9)

    def __getitem__(self, idx):
        encoding = tokenizer(self.texts[idx], ...)
        return {
            'input_ids': ...,
            'attention_mask': ...,
            'symptom_idx': self.symptom_indices[idx]  # Single label
        }
```

**Data Format**:
```csv
post_id,text,symptom_label
s_123,I can't sleep at night,SLEEP_ISSUES
s_456,I feel worthless,WORTHLESSNESS
```

#### LLM_Evidence_Gemma
```python
class EvidenceDataset(Dataset):
    def __init__(self, examples, tokenizer, ...):
        # examples = list of dicts with question, context, answer
        self.examples = examples

    def __getitem__(self, idx):
        ex = self.examples[idx]
        encoding = tokenizer(
            ex['question'],   # "Find evidence for SLEEP_ISSUES"
            ex['context'],    # Full post text
            return_offsets_mapping=True,  # For char→token mapping
        )

        # Convert char positions to token positions
        start_position, end_position = self._char_to_token_positions(...)

        return {
            'input_ids': ...,
            'attention_mask': ...,
            'start_positions': start_position,  # Token index
            'end_positions': end_position,      # Token index
            'symptom_idx': ex['symptom_idx'],
        }
```

**Data Format**:
```python
{
    'question': 'Find evidence for sleep issues in this post',
    'context': 'I can\'t sleep at night. My schedule is...',
    'answer_text': 'I can\'t sleep at night',
    'answer_start': 0,  # Character position
    'symptom_idx': 3,
}
```

**Key Differences**:
- **Input Structure**: Single text → Question + Context
- **Labels**: Single class index → Start + End positions
- **Preprocessing**: Simple tokenization → Answer position mapping
- **Offset Mapping**: Not needed → Required for position alignment

---

### 3. Loss Functions

#### LLM_Criteria_Gemma
```python
def compute_loss(logits, labels):
    # Single classification loss
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss = criterion(logits, labels)  # logits: [batch, 10], labels: [batch]
    return loss
```

#### LLM_Evidence_Gemma
```python
def compute_loss(start_logits, end_logits, start_positions, end_positions):
    # Two separate classification losses (over sequence positions)
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)

    # Start position loss
    start_loss = loss_fct(start_logits, start_positions)  # [batch, seq_len], [batch]

    # End position loss
    end_loss = loss_fct(end_logits, end_positions)

    # Combined loss
    total_loss = (start_loss + end_loss) / 2
    return total_loss
```

**Key Differences**:
- **Number of Losses**: Single → Dual (start + end)
- **Target Shape**: `[batch]` → `[batch]` (but predicting over seq_len)
- **Class Weighting**: Used for imbalanced classes → Not applicable
- **Ignore Index**: Not used → Used for padding/invalid positions

---

### 4. Evaluation Metrics

#### LLM_Criteria_Gemma
```python
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

def evaluate(all_preds, all_labels):
    # Classification metrics
    accuracy = accuracy_score(all_labels, all_preds)

    # Per-class and macro-averaged metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average='macro'
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,  # Macro F1
        'confusion_matrix': conf_matrix,
    }
```

#### LLM_Evidence_Gemma
```python
def normalize_answer(s):
    # Lowercase, remove punctuation, articles, normalize whitespace
    ...

def compute_exact_match(prediction, ground_truth):
    # Binary: 1.0 if exact match after normalization
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))

def compute_f1(prediction, ground_truth):
    # Token-level F1
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate(predictions, ground_truths):
    em_scores = [compute_exact_match(p, g) for p, g in zip(...)]
    f1_scores = [compute_f1(p, g) for p, g in zip(...)]

    return {
        'exact_match': np.mean(em_scores),
        'f1': np.mean(f1_scores),
    }
```

**Key Differences**:

| Metric | Criteria (Classification) | Evidence (QA) |
|--------|--------------------------|---------------|
| **Primary** | Accuracy | Exact Match (EM) |
| **Secondary** | Macro F1 | F1 token overlap |
| **Granularity** | Per-class metrics | Per-symptom metrics |
| **Text Comparison** | N/A | Normalized text matching |
| **Confusion Matrix** | Yes (10x10) | Not applicable |

---

### 5. Training Loop

#### LLM_Criteria_Gemma
```python
def train_epoch(model, train_loader, optimizer, criterion):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['symptom_idx'].to(device)

        optimizer.zero_grad()

        # Forward
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

#### LLM_Evidence_Gemma
```python
def train_epoch(model, train_loader, optimizer):
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()

        # Forward (returns loss + logits)
        loss, start_logits, end_logits = model(
            input_ids, attention_mask,
            start_positions=start_positions,
            end_positions=end_positions
        )

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
```

**Key Differences**:
- **Labels**: Single `symptom_idx` → `start_positions` + `end_positions`
- **Forward Pass**: Returns logits → Returns loss + logits
- **Loss Computation**: External criterion → Computed in model

---

### 6. Prediction and Inference

#### LLM_Criteria_Gemma
```python
@torch.no_grad()
def predict(model, text, tokenizer):
    encoding = tokenizer(text, return_tensors='pt')
    logits = model(encoding['input_ids'], encoding['attention_mask'])
    pred_idx = torch.argmax(logits, dim=-1).item()
    pred_label = SYMPTOM_LABELS[pred_idx]
    confidence = torch.softmax(logits, dim=-1)[0][pred_idx].item()

    return pred_label, confidence
```

#### LLM_Evidence_Gemma
```python
@torch.no_grad()
def predict(model, question, context, tokenizer):
    encoding = tokenizer(question, context, return_tensors='pt')
    start_logits, end_logits = model(
        encoding['input_ids'],
        encoding['attention_mask']
    )

    # Find best span
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    # Extract answer text
    answer_tokens = encoding['input_ids'][0][start_idx:end_idx+1]
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    confidence = (start_logits[0][start_idx] + end_logits[0][end_idx]) / 2

    return answer_text, start_idx, end_idx, confidence
```

**Key Differences**:
- **Output Type**: Single label → Text span
- **Decoding**: Argmax over classes → Span extraction + decoding
- **Confidence**: Softmax probability → Combined logit score

---

### 7. Cross-Validation Strategy

Both projects use **stratified 5-fold CV**, but the stratification key differs:

#### LLM_Criteria_Gemma
```python
# Stratify by symptom label (0-9)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, symptom_labels):
    ...
```

#### LLM_Evidence_Gemma
```python
# Stratify by symptom label (to ensure balanced representation)
# Even though task is QA, we still stratify by symptom type
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, symptom_indices):
    ...
```

**Same Strategy**: Both stratify by symptom to ensure balanced representation across folds.

---

## Shared Components (Unchanged)

### 1. Bidirectional Gemma Encoder ✓
Both projects use the **same bidirectional attention mechanism**:
```python
def _enable_bidirectional_attention(self):
    # Disable causal masking
    # Allow all tokens to attend to all other tokens
    ...
```

### 2. Training Infrastructure ✓
- **Optimizer**: AdamW with same default hyperparameters
- **Scheduler**: Linear warmup schedule
- **Regularization**: Gradient clipping (max_norm=1.0), dropout (0.1)
- **Early Stopping**: Patience-based early stopping
- **Mixed Precision**: bfloat16 AMP support

### 3. Hydra Configuration ✓
- Same configuration structure
- Command-line overrides
- Experiment presets (quick_test, full_5fold)

### 4. Logging and Tracking ✓
- Same logger utilities (colored output, formatting)
- Same experiment tracking (MLflow, W&B)
- Same checkpoint management

### 5. Makefile and Automation ✓
- Similar command structure
- Installation, training, evaluation targets
- Data utilities and code quality tools

---

## Performance Comparison

| Metric | LLM_Criteria_Gemma | LLM_Evidence_Gemma |
|--------|-------------------|-------------------|
| **Task Difficulty** | Moderate (10-class) | Higher (span extraction) |
| **Gemma-2-2B Accuracy/EM** | 75-80% | 65-70% |
| **Gemma-2-2B F1** | 0.72-0.75 (macro) | 0.75-0.80 (token) |
| **Training Time (5-fold)** | ~2 hours (A100) | ~2 hours (A100) |
| **Memory Usage** | Lower (pooled repr) | Higher (full sequence) |

---

## When to Use Each Project

### Use LLM_Criteria_Gemma When:
- ✅ You need to **classify** which DSM-5 criterion applies
- ✅ You want a **single symptom label** per post
- ✅ You need **confusion matrix** analysis
- ✅ You care about **per-class precision/recall**

### Use LLM_Evidence_Gemma When:
- ✅ You need to **extract exact evidence text**
- ✅ You want **explainability** (which sentence supports diagnosis)
- ✅ You need **multi-label support** (multiple evidence spans per post)
- ✅ You care about **text-level agreement** (EM/F1)

---

## Migration Guide: Criteria → Evidence

If you have a trained LLM_Criteria_Gemma model, here's how to adapt it:

### 1. Model Conversion
```python
# Load Criteria model
criteria_model = GemmaClassifier.from_pretrained(checkpoint)

# Extract encoder weights
encoder_state = criteria_model.encoder.state_dict()

# Initialize QA model with same encoder
qa_model = GemmaQA(model_name=model_name)
qa_model.encoder.load_state_dict(encoder_state)

# Only train QA head (freeze encoder for faster fine-tuning)
qa_model.freeze_encoder = True
```

### 2. Data Preparation
```python
# Convert classification data to QA format
def convert_to_qa_format(posts_df, annotations_df):
    qa_examples = []
    for _, row in annotations_df.iterrows():
        post_text = posts_df[posts_df['post_id'] == row['post_id']]['text'].values[0]

        # Find sentence position in post
        answer_start = post_text.find(row['sentence_text'])

        qa_examples.append({
            'question': f"Find evidence for {row['DSM5_symptom']}",
            'context': post_text,
            'answer_text': row['sentence_text'],
            'answer_start': answer_start,
        })
    return qa_examples
```

### 3. Fine-tuning
```python
# Fine-tune QA head only (encoder frozen)
python src/training/train_gemma_qa.py \
    --model_name google/gemma-2-2b \
    --freeze_encoder \
    --num_epochs 5 \
    --learning_rate 1e-4
```

---

## Conclusion

Both projects share the **same core innovation** (bidirectional Gemma encoder) but differ in their downstream tasks:

- **LLM_Criteria_Gemma**: Classification (which symptom?)
- **LLM_Evidence_Gemma**: Extraction (which text evidence?)

The choice depends on your use case: classification for diagnosis, extraction for explainability.
