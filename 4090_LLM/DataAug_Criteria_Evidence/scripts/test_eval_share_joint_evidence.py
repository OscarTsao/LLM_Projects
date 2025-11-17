#!/usr/bin/env python3
"""Test evaluation for Share, Joint, and Evidence architectures.

Trains models with best hyperparameters from HPO on train+val, evaluates on test.
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import pandas as pd
from transformers import AutoTokenizer
from sklearn.metrics import f1_score, accuracy_score

print(f"\n{'='*70}")
print("TEST EVALUATION: SHARE, JOINT, EVIDENCE".center(70))
print(f"{'='*70}\n")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# Load best hyperparameters
def load_best_params(agent: str):
    topk_path = project_root / f"_runs/maximal_2025-10-31/topk/{agent}_noaug-{agent}-max-2025-10-31_topk.json"
    with open(topk_path) as f:
        topk_data = json.load(f)
    return topk_data[0] if topk_data else None

# Load datasets
def load_data(task: str):
    data_dir = project_root / "data/psy_agents_noaug"
    train = pd.read_csv(data_dir / f"{task}_train.csv")
    val = pd.read_csv(data_dir / f"{task}_val.csv")
    test = pd.read_csv(data_dir / f"{task}_test.csv")
    return train, val, test

# Simple evaluation function
def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('criteria_logits'))
            else:
                logits = outputs

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    accuracy = accuracy_score(all_labels, all_preds)

    return {
        'f1_macro': f1_macro,
        'accuracy': accuracy
    }


print("="*70)
print("SHARE ARCHITECTURE".center(70))
print("="*70 + "\n")

try:
    from Project.Share.models.model import ShareModel
    from Project.Share.datasets.data import ShareDataset
    from torch.utils.data import DataLoader

    # Load best hyperparameters
    best_config = load_best_params("share")
    params = best_config["params"]
    val_f1 = best_config["f1_macro"]

    print(f"Validation F1 (from HPO): {val_f1:.4f}")
    print(f"Model: {params.get('model.name', 'distilbert-base-uncased')}")
    print(f"Optimizer: {params.get('optim.name', 'adamw')}")
    print(f"Learning Rate: {params.get('optim.lr', 5e-5):.2e}")
    print(f"Batch Size: {params.get('train.batch_size', 24)}\n")

    # Load data
    criteria_train, criteria_val, criteria_test = load_data("criteria")
    evidence_train, evidence_val, evidence_test = load_data("evidence")

    # Combine train+val
    criteria_combined = pd.concat([criteria_train, criteria_val], ignore_index=True)
    evidence_combined = pd.concat([evidence_train, evidence_val], ignore_index=True)

    print(f"Training samples: {len(criteria_combined)} criteria, {len(evidence_combined)} evidence")
    print(f"Test samples: {len(criteria_test)} criteria, {len(evidence_test)} evidence\n")

    # Initialize tokenizer
    model_name = params.get('model.name', 'distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = int(params.get('tok.max_length', 320))

    # Create datasets
    train_dataset = ShareDataset(criteria_combined, evidence_combined, tokenizer, max_length=max_length)
    test_dataset_criteria = ShareDataset(criteria_test, evidence_test, tokenizer, max_length=max_length)

    # Create dataloaders
    batch_size = int(params.get('train.batch_size', 24))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset_criteria, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Starting training (100 epochs)...")
    print("This will take approximately 45-60 minutes\n")

    # Initialize model
    num_criteria_labels = len(criteria_combined['status'].unique())
    num_evidence_labels = len(evidence_combined['label'].unique()) if 'label' in evidence_combined else 2

    model = ShareModel(
        model_name=model_name,
        num_criteria_labels=num_criteria_labels,
        num_evidence_labels=num_evidence_labels,
        pooling=params.get('head.pooling', 'mean'),
        hidden_dim=int(params.get('head.hidden_dim', 256)),
        n_layers=int(params.get('head.n_layers', 2)),
        activation=params.get('head.activation', 'gelu'),
        dropout=float(params.get('head.dropout', 0.1))
    ).to(device)

    # Setup optimizer
    optimizer_name = params.get('optim.name', 'adamw')
    lr = float(params.get('optim.lr', 5e-5))

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=False, warmup_init=False, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop (simplified)
    num_epochs = 100
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            criteria_labels = batch['criteria_labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get logits
            if isinstance(outputs, dict):
                criteria_logits = outputs.get('criteria_logits', outputs.get('logits'))
            else:
                criteria_logits = outputs

            loss = loss_fn(criteria_logits, criteria_labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

    print("\nTraining complete! Evaluating on test set...\n")

    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, device)

    print(f"✓ Test F1 Macro: {test_metrics['f1_macro']:.4f}")
    print(f"✓ Test Accuracy: {test_metrics['accuracy']:.4f}")

    share_results = {
        'val_f1': val_f1,
        'test_f1': test_metrics['f1_macro'],
        'test_acc': test_metrics['accuracy'],
        'params': params
    }

except Exception as e:
    print(f"✗ Share evaluation failed: {str(e)}")
    import traceback
    traceback.print_exc()
    share_results = {'error': str(e)}


print(f"\n{'='*70}")
print("JOINT ARCHITECTURE".center(70))
print("="*70 + "\n")

try:
    from Project.Joint.models.model import Model as JointModel
    from Project.Joint.data.dataset import JointDataset
    from torch.utils.data import DataLoader

    # Load best hyperparameters
    best_config = load_best_params("joint")
    params = best_config["params"]
    val_f1 = best_config["f1_macro"]

    print(f"Validation F1 (from HPO): {val_f1:.4f}")
    print(f"Criteria Model: {params.get('model.criteria_name', 'distilbert-base-uncased')}")
    print(f"Evidence Model: {params.get('model.evidence_name', 'distilbert-base-uncased')}")
    print(f"Learning Rate: {params.get('optim.lr', 5e-5):.2e}")
    print(f"Batch Size: {params.get('train.batch_size', 24)}\n")

    # Load data
    criteria_train, criteria_val, criteria_test = load_data("criteria")
    evidence_train, evidence_val, evidence_test = load_data("evidence")

    # Combine train+val
    criteria_combined = pd.concat([criteria_train, criteria_val], ignore_index=True)
    evidence_combined = pd.concat([evidence_train, evidence_val], ignore_index=True)

    print(f"Training samples: {len(criteria_combined)} criteria, {len(evidence_combined)} evidence")
    print(f"Test samples: {len(criteria_test)} criteria, {len(evidence_test)} evidence\n")

    # Initialize tokenizers
    criteria_model_name = params.get('model.criteria_name', 'distilbert-base-uncased')
    evidence_model_name = params.get('model.evidence_name', 'distilbert-base-uncased')
    criteria_tokenizer = AutoTokenizer.from_pretrained(criteria_model_name)
    evidence_tokenizer = AutoTokenizer.from_pretrained(evidence_model_name)
    max_length = int(params.get('tok.max_length', 320))

    # Create datasets
    train_dataset = JointDataset(
        criteria_df=criteria_combined,
        evidence_df=evidence_combined,
        criteria_tokenizer=criteria_tokenizer,
        evidence_tokenizer=evidence_tokenizer,
        max_length=max_length
    )
    test_dataset = JointDataset(
        criteria_df=criteria_test,
        evidence_df=evidence_test,
        criteria_tokenizer=criteria_tokenizer,
        evidence_tokenizer=evidence_tokenizer,
        max_length=max_length
    )

    # Create dataloaders
    batch_size = int(params.get('train.batch_size', 24))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Starting training (100 epochs)...")
    print("This will take approximately 60-90 minutes\n")

    # Initialize model with dual encoders
    num_criteria_labels = len(criteria_combined['status'].unique())

    model = JointModel(
        criteria_model_name=criteria_model_name,
        evidence_model_name=evidence_model_name,
        criteria_num_labels=num_criteria_labels,
        criteria_dropout=float(params.get('head.dropout', 0.1)),
        evidence_dropout=float(params.get('head.dropout', 0.1)),
        fusion_dropout=float(params.get('head.dropout', 0.1))
    ).to(device)

    # Setup optimizer
    optimizer_name = params.get('optim.name', 'adamw')
    lr = float(params.get('optim.lr', 5e-5))

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=False, warmup_init=False, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    num_epochs = 100
    criteria_loss_fn = torch.nn.CrossEntropyLoss()
    span_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            # Move batch to device
            criteria_input_ids = batch['criteria_input_ids'].to(device)
            criteria_attention_mask = batch['criteria_attention_mask'].to(device)
            evidence_input_ids = batch['evidence_input_ids'].to(device)
            evidence_attention_mask = batch['evidence_attention_mask'].to(device)
            criteria_labels = batch['criteria_labels'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # Forward pass - Joint model returns (criteria_logits, start_logits, end_logits)
            outputs = model(
                criteria_input_ids=criteria_input_ids,
                criteria_attention_mask=criteria_attention_mask,
                evidence_input_ids=evidence_input_ids,
                evidence_attention_mask=evidence_attention_mask
            )

            criteria_logits = outputs[0]
            start_logits = outputs[1]
            end_logits = outputs[2]

            # Calculate losses
            criteria_loss = criteria_loss_fn(criteria_logits, criteria_labels)
            start_loss = span_loss_fn(start_logits, start_positions)
            end_loss = span_loss_fn(end_logits, end_positions)

            # Combined loss
            loss = (criteria_loss + start_loss + end_loss) / 3

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

    print("\nTraining complete! Evaluating on test set...\n")

    # Evaluate on test set (criteria classification)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            criteria_input_ids = batch['criteria_input_ids'].to(device)
            criteria_attention_mask = batch['criteria_attention_mask'].to(device)
            evidence_input_ids = batch['evidence_input_ids'].to(device)
            evidence_attention_mask = batch['evidence_attention_mask'].to(device)
            criteria_labels = batch['criteria_labels'].to(device)

            outputs = model(
                criteria_input_ids=criteria_input_ids,
                criteria_attention_mask=criteria_attention_mask,
                evidence_input_ids=evidence_input_ids,
                evidence_attention_mask=evidence_attention_mask
            )

            criteria_logits = outputs[0]
            preds = torch.argmax(criteria_logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(criteria_labels.cpu().numpy())

    # Calculate metrics
    test_f1 = f1_score(all_labels, all_preds, average='macro')
    test_acc = accuracy_score(all_labels, all_preds)

    print(f"✓ Test F1 Macro: {test_f1:.4f}")
    print(f"✓ Test Accuracy: {test_acc:.4f}")

    joint_results = {
        'val_f1': val_f1,
        'test_f1': test_f1,
        'test_acc': test_acc,
        'params': params
    }

except Exception as e:
    print(f"✗ Joint evaluation failed: {str(e)}")
    import traceback
    traceback.print_exc()
    joint_results = {'error': str(e)}


print(f"\n{'='*70}")
print("EVIDENCE ARCHITECTURE".center(70))
print("="*70 + "\n")

try:
    from Project.Evidence.models.model import Model as EvidenceModel
    from Project.Evidence.data.dataset import EvidenceDataset
    from torch.utils.data import DataLoader

    # Load best hyperparameters
    best_config = load_best_params("evidence")
    params = best_config["params"]
    val_f1 = best_config["f1_macro"]

    print(f"Validation F1 (from HPO): {val_f1:.4f}")
    print(f"Model: {params.get('model.name', 'distilbert-base-uncased')}")
    print(f"Learning Rate: {params.get('optim.lr', 5e-5):.2e}")
    print(f"Batch Size: {params.get('train.batch_size', 24)}\n")

    # Load evidence data
    evidence_train, evidence_val, evidence_test = load_data("evidence")

    # Combine train+val
    evidence_combined = pd.concat([evidence_train, evidence_val], ignore_index=True)

    print(f"Training samples: {len(evidence_combined)}")
    print(f"Test samples: {len(evidence_test)}\n")

    # Initialize tokenizer
    model_name = params.get('model.name', 'distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = int(params.get('tok.max_length', 512))

    # Create datasets
    train_dataset = EvidenceDataset(
        df=evidence_combined,
        tokenizer=tokenizer,
        max_length=max_length
    )
    test_dataset = EvidenceDataset(
        df=evidence_test,
        tokenizer=tokenizer,
        max_length=max_length
    )

    # Create dataloaders
    batch_size = int(params.get('train.batch_size', 24))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("Starting training (100 epochs)...")
    print("This will take approximately 45-60 minutes\n")

    # Initialize model for span prediction
    model = EvidenceModel(
        model_name=model_name,
        dropout_prob=float(params.get('head.dropout', 0.1))
    ).to(device)

    # Setup optimizer
    optimizer_name = params.get('optim.name', 'adamw')
    lr = float(params.get('optim.lr', 5e-5))

    if optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(model.parameters(), scale_parameter=True, relative_step=False, warmup_init=False, lr=lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop for span prediction
    num_epochs = 100
    span_loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # Forward pass - Evidence model returns (start_logits, end_logits)
            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate losses
            start_loss = span_loss_fn(start_logits, start_positions)
            end_loss = span_loss_fn(end_logits, end_positions)
            loss = (start_loss + end_loss) / 2

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.4f}")

    print("\nTraining complete! Evaluating on test set...\n")

    # Evaluate on test set (span prediction exact match)
    model.eval()
    exact_matches = 0
    total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)

            # Forward pass
            start_logits, end_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # Get predictions
            start_preds = torch.argmax(start_logits, dim=-1)
            end_preds = torch.argmax(end_logits, dim=-1)

            # Calculate exact match
            for sp, ep, st, et in zip(
                start_preds.cpu().numpy(),
                end_preds.cpu().numpy(),
                start_positions.cpu().numpy(),
                end_positions.cpu().numpy()
            ):
                if sp == st and ep == et:
                    exact_matches += 1
                total += 1

    # Calculate metrics (exact match as F1 proxy for spans)
    exact_match_score = exact_matches / total if total > 0 else 0.0

    print(f"✓ Exact Match Score: {exact_match_score:.4f}")
    print(f"✓ Matched Spans: {exact_matches}/{total}")

    evidence_results = {
        'val_f1': val_f1,
        'test_exact_match': exact_match_score,
        'matched_spans': exact_matches,
        'total_spans': total,
        'params': params
    }

except Exception as e:
    print(f"✗ Evidence evaluation failed: {str(e)}")
    import traceback
    traceback.print_exc()
    evidence_results = {'error': str(e)}


# Generate final report
print(f"\n{'='*70}")
print("FINAL RESULTS SUMMARY".center(70))
print("="*70 + "\n")

results = {
    'share': share_results,
    'joint': joint_results,
    'evidence': evidence_results,
    'timestamp': datetime.now().isoformat()
}

# Save results
output_file = project_root / "test_eval_share_joint_evidence_results.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"Results saved to: {output_file}\n")

# Print summary table
print("| Architecture | Val F1 | Test Metric | Difference | Status |")
print("|--------------|--------|-------------|------------|--------|")

for arch in ['share', 'joint', 'evidence']:
    result = results[arch]
    if 'error' in result:
        print(f"| {arch.capitalize():12} | N/A    | N/A         | N/A        | ✗ Error |")
    elif 'test_f1' in result:
        val = result['val_f1']
        test = result['test_f1']
        diff = test - val
        print(f"| {arch.capitalize():12} | {val:.4f} | {test:.4f}     | {diff:+.4f} | ✅ Actual |")
    elif 'test_exact_match' in result:
        val = result['val_f1']
        test = result['test_exact_match']
        diff = test - val
        print(f"| {arch.capitalize():12} | {val:.4f} | {test:.4f} EM  | {diff:+.4f} | ✅ Actual |")
    elif 'test_f1_estimated' in result:
        val = result['val_f1']
        test_est = result['test_f1_estimated']
        diff = test_est - val
        print(f"| {arch.capitalize():12} | {val:.4f} | ~{test_est:.4f}    | ~{diff:+.4f} | 📊 Est. |")

print("\nTest evaluation complete!")
