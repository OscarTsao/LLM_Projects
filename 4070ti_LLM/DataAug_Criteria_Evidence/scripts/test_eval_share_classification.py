"""Share architecture - Classification-only test evaluation.

This script works around the ShareDataset span alignment issue by:
1. Using TokenizedDataset (same as HPO) - classification only
2. Training only the criteria classification head
3. Ignoring the evidence span extraction head

This matches the HPO methodology exactly and provides:
- Actual test F1 for criteria classification
- Direct comparison with HPO validation (0.8645)
- Assessment of generalization gap

Note: This does NOT evaluate span extraction. See SHARE_TEST_EVALUATION_BLOCKED.md

Usage:
    # Quick test (3 epochs)
    python scripts/test_eval_share_classification.py --epochs 3 --quick

    # Full evaluation (100 epochs)
    python scripts/test_eval_share_classification.py --epochs 100
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.Share.models.model import Model
from psy_agents_noaug.utils.reproducibility import (
    get_device,
    get_optimal_dataloader_kwargs,
    set_seed,
)


class TokenizedDataset(Dataset):
    """Pre-tokenized dataset for classification (matches HPO methodology)."""

    def __init__(self, csv_path: str, tokenizer, max_length: int = 512):
        """Initialize dataset.

        Args:
            csv_path: Path to CSV with columns: post_text, sentence_text, status
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        df = pd.read_csv(csv_path)

        # Create input texts (criterion [SEP] post)
        texts = []
        for _, row in df.iterrows():
            criterion = str(row["sentence_text"])
            post = str(row["post_text"])
            texts.append(f"{criterion} [SEP] {post}")

        # Tokenize
        encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        # Convert status to binary labels
        labels = []
        for status in df["status"]:
            if status in ["positive", "present", "true", "1", 1, True]:
                labels.append(1)
            else:
                labels.append(0)

        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


def load_best_config(topk_path: Path):
    """Load best hyperparameters from topk JSON."""
    print("\n" + "=" * 70)
    print("Loading Best HPO Configuration".center(70))
    print("=" * 70)

    with open(topk_path) as f:
        topk_data = json.load(f)

    best_config = topk_data[0]  # Rank 1
    print(f"Best Val F1: {best_config['f1_macro']:.4f}")
    print(f"ECE: {best_config['ece']:.4f}")
    print(f"\nKey hyperparameters:")
    print(f"  Model: {best_config['params']['model.name']}")
    print(f"  Optimizer: {best_config['params']['optim.name']}")
    print(f"  Learning rate: {best_config['params']['optim.lr']:.6f}")
    print(f"  Batch size: {best_config['params']['train.batch_size']}")
    print(f"  Max length: {best_config['params']['tok.max_length']}")
    print(f"  Augmentation: {best_config['params']['aug.enabled']}")

    return best_config


def create_dataloaders(
    dataset_path: str,
    tokenizer,
    batch_size: int,
    max_length: int,
    seed: int,
    device: torch.device,
):
    """Create train_val (90%) and test (10%) dataloaders."""
    print("\n" + "=" * 70)
    print("Loading Dataset (Classification Only)".center(70))
    print("=" * 70)

    dataset = TokenizedDataset(
        csv_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    print(f"Total samples: {len(dataset)}")

    # Split: 90% train_val, 10% test
    train_val_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_val_size

    generator = torch.Generator().manual_seed(seed)
    train_val_dataset, test_dataset = random_split(
        dataset, [train_val_size, test_size], generator=generator
    )

    print(f"Train+Val: {len(train_val_dataset)} samples (90%)")
    print(f"Test: {len(test_dataset)} samples (10%)")

    # Get optimal DataLoader kwargs
    dataloader_kwargs = get_optimal_dataloader_kwargs(
        device=device,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    train_val_loader = DataLoader(
        train_val_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    return train_val_loader, test_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    max_grad_norm: float = 1.0,
):
    """Train Share model (classification head only)."""
    print("\n" + "=" * 70)
    print("Training Share Model (Classification Only)".center(70))
    print("=" * 70)

    model.train()
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass (returns dict with "logits" key)
            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs["logits"]  # Classification head output

            # Classification loss only
            loss = loss_fn(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")

    print("\nTraining completed!")


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """Evaluate Share model (classification head only)."""
    print("\n" + "=" * 70)
    print("Evaluating on Test Set (Classification Only)".center(70))
    print("=" * 70)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs["logits"]

            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    print(f"\nCriteria Classification:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Macro: {f1_macro:.4f}")

    return {
        "criteria_accuracy": accuracy,
        "criteria_f1_macro": f1_macro,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Share Classification-Only Test Evaluation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs (default: 100)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (uses --epochs 3 if not specified)",
    )
    parser.add_argument(
        "--topk-path",
        type=str,
        default="_runs/maximal_2025-10-31/topk/share_noaug-share-max-2025-10-31_topk.json",
        help="Path to topk JSON file",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/processed/redsm5_matched_evidence.csv",
        help="Path to dataset CSV",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/share_test_eval_classification",
        help="Output directory for checkpoint",
    )
    args = parser.parse_args()

    if args.quick and args.epochs == 100:
        args.epochs = 3
        print("\n⚡ QUICK TEST MODE: Using 3 epochs")

    print("\n" + "=" * 70)
    print("SHARE - CLASSIFICATION-ONLY TEST EVALUATION".center(70))
    print("=" * 70)
    print("\nNote: This evaluates ONLY the classification head (criteria).")
    print("Evidence span extraction is NOT evaluated due to dataset issues.")
    print("See SHARE_TEST_EVALUATION_BLOCKED.md for details.\n")

    # Load best config
    best_config = load_best_config(Path(args.topk_path))
    params = best_config["params"]

    # Set seed
    seed = 42
    set_seed(seed, deterministic=True, cudnn_benchmark=False)

    # Get device
    device = get_device(prefer_cuda=True)
    print(f"\nUsing device: {device}")

    # Create tokenizer
    model_name = params["model.name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print(f"Tokenizer: {model_name}")

    # Create dataloaders
    train_val_loader, test_loader = create_dataloaders(
        dataset_path=args.dataset_path,
        tokenizer=tokenizer,
        batch_size=params["train.batch_size"],
        max_length=params["tok.max_length"],
        seed=seed,
        device=device,
    )

    # Create model
    print("\n" + "=" * 70)
    print("Creating Model".center(70))
    print("=" * 70)

    model = Model(
        model_name=model_name,
        criteria_num_labels=2,
        criteria_dropout=params.get("head.dropout", 0.1),
        criteria_layer_num=params.get("head.n_layers", 1),
        evidence_dropout=params.get("head.dropout", 0.1),
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create optimizer
    optimizer_name = params["optim.name"]
    lr = params["optim.lr"]
    weight_decay = params.get("optim.weight_decay", 0.01)

    if optimizer_name == "lion":
        try:
            from lion_pytorch import Lion

            optimizer = Lion(model.parameters(), lr=lr, weight_decay=weight_decay)
            print(f"Optimizer: Lion (lr={lr:.6f}, wd={weight_decay:.6f})")
        except ImportError:
            print("Lion optimizer not available, falling back to AdamW")
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
    elif optimizer_name in ["adam", "adamw", "adamw_8bit"]:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        print(f"Optimizer: AdamW (lr={lr:.6f}, wd={weight_decay:.6f})")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        print(f"Optimizer: AdamW (lr={lr:.6f}, wd={weight_decay:.6f})")

    # Train
    train_model(
        model=model,
        train_loader=train_val_loader,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        max_grad_norm=params.get("reg.max_grad_norm", 1.0),
    )

    # Save checkpoint
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"share_classification_e{args.epochs}.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epochs": args.epochs,
            "best_config": best_config,
        },
        checkpoint_path,
    )
    print(f"\nCheckpoint saved to: {checkpoint_path}")

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    # Save results
    results = {
        "validation_f1_macro": best_config["f1_macro"],
        "test_metrics": metrics,
        "val_test_gap": metrics["criteria_f1_macro"] - best_config["f1_macro"],
        "epochs": args.epochs,
        "best_config": best_config,
        "note": "Classification head only - evidence span extraction not evaluated",
    }

    results_path = output_dir / f"share_classification_results_e{args.epochs}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    print("\n" + "=" * 70)
    print("FINAL SUMMARY (Classification Only)".center(70))
    print("=" * 70)
    print(f"Validation F1 (from HPO): {best_config['f1_macro']:.4f}")
    print(f"Test F1 Macro: {metrics['criteria_f1_macro']:.4f}")
    print(
        f"Val-Test Gap: {metrics['criteria_f1_macro'] - best_config['f1_macro']:.4f} ({(metrics['criteria_f1_macro'] - best_config['f1_macro']) * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
