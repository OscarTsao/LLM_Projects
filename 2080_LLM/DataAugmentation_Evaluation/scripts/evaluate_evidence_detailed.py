"""
Comprehensive evaluation script for evidence binding agent.

This script:
1. Loads a trained evidence binding model
2. Evaluates on validation and test sets
3. Saves detailed predictions to CSV with post_id, post, predicted sentences, ground truth sentences
4. Computes and saves comprehensive evaluation metrics
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.agents.evidence_binding import EvidenceBindingAgent, EvidenceBindingConfig
from src.data.evidence_loader import EvidenceCollator, EvidenceDataset, load_evidence_annotations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: str, config: Dict) -> Tuple[EvidenceBindingAgent, torch.device]:
    """Load the evidence binding model from checkpoint."""

    # Create agent config
    agent_config = EvidenceBindingConfig(
        model_name=config.get("model_name", "microsoft/deberta-v3-base"),
        max_seq_length=config.get("max_seq_length", 512),
        dropout=config.get("dropout", 0.1),
        label_smoothing=config.get("label_smoothing", 0.0),
        max_span_length=config.get("max_span_length", 50),
        span_threshold=config.get("span_threshold", 0.5),
    )

    # Create agent
    agent = EvidenceBindingAgent(agent_config)

    # Load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle compiled model state dict
    checkpoint_is_compiled = any(k.startswith("_orig_mod.") for k in state_dict.keys())
    model_is_compiled = hasattr(agent, "_orig_mod")

    if checkpoint_is_compiled and not model_is_compiled:
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    elif not checkpoint_is_compiled and model_is_compiled:
        state_dict = {f"_orig_mod.{k}": v for k, v in state_dict.items()}

    agent.load_state_dict(state_dict)
    agent.to(device)
    agent.eval()

    logger.info(f"Model loaded from {checkpoint_path}")
    logger.info(f"Using device: {device}")

    return agent, device


def tokens_to_text(input_ids: torch.Tensor, tokenizer: AutoTokenizer, span: Tuple[int, int]) -> str:
    """Convert token span to text."""
    start, end = span
    tokens = input_ids[start:end+1]
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    return text.strip()


def evaluate_dataset(
    agent: EvidenceBindingAgent,
    dataset: EvidenceDataset,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    output_dir: Path
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Evaluate the evidence binding agent on a dataset.

    Returns:
        metrics: Dictionary of evaluation metrics
        predictions_df: DataFrame with detailed predictions
    """
    logger.info(f"Evaluating on {split_name} set...")

    tokenizer = agent.tokenizer

    all_predictions = []
    all_start_labels = []
    all_end_labels = []
    all_start_probs = []
    all_end_probs = []

    prediction_records = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {split_name}")):
            # Get original examples for this batch
            batch_start_idx = batch_idx * dataloader.batch_size
            batch_end_idx = min(batch_start_idx + dataloader.batch_size, len(dataset))
            batch_examples = dataset.examples[batch_start_idx:batch_end_idx]

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            # Forward pass
            outputs = agent.predict(input_ids=input_ids, attention_mask=attention_mask)

            # Extract predictions
            predicted_spans = outputs.predictions  # List of lists of (start, end) tuples
            start_probs = outputs.probabilities["start"]
            end_probs = outputs.probabilities["end"]

            # Process each example in batch
            for i in range(len(batch_examples)):
                example = batch_examples[i]

                # Get ground truth spans
                gt_spans = [(span.start_token, span.end_token) for span in example.evidence_spans]

                # Get predicted spans for this example
                pred_spans = predicted_spans[i] if i < len(predicted_spans) else []

                # Convert spans to text
                example_input_ids = input_ids[i].cpu()

                pred_texts = []
                for span in pred_spans:
                    try:
                        text = tokens_to_text(example_input_ids, tokenizer, span)
                        pred_texts.append(text)
                    except:
                        pred_texts.append("")

                gt_texts = []
                for span in gt_spans:
                    try:
                        text = tokens_to_text(example_input_ids, tokenizer, span)
                        gt_texts.append(text)
                    except:
                        gt_texts.append("")

                # Store prediction record
                prediction_records.append({
                    "post_id": example.post_id,
                    "post": example.post_text,
                    "criterion": example.criterion,
                    "has_evidence": example.has_evidence,
                    "num_predicted_spans": len(pred_spans),
                    "num_ground_truth_spans": len(gt_spans),
                    "predicted_spans": str(pred_spans),
                    "ground_truth_spans": str(gt_spans),
                    "predicted_sentences": " | ".join(pred_texts),
                    "ground_truth_sentences": " | ".join(gt_texts),
                })

                # Collect token-level predictions for metrics
                seq_len = attention_mask[i].sum().item()
                start_prob = start_probs[i, :seq_len].cpu().numpy()
                end_prob = end_probs[i, :seq_len].cpu().numpy()
                start_label = start_positions[i, :seq_len].cpu().numpy()
                end_label = end_positions[i, :seq_len].cpu().numpy()

                all_start_probs.extend(start_prob)
                all_end_probs.extend(end_prob)
                all_start_labels.extend(start_label)
                all_end_labels.extend(end_label)

    # Convert token probabilities to binary predictions
    all_start_preds = [1 if p > agent.config.span_threshold else 0 for p in all_start_probs]
    all_end_preds = [1 if p > agent.config.span_threshold else 0 for p in all_end_probs]

    # Compute metrics
    start_accuracy = accuracy_score(all_start_labels, all_start_preds)
    start_precision = precision_score(all_start_labels, all_start_preds, zero_division=0)
    start_recall = recall_score(all_start_labels, all_start_preds, zero_division=0)
    start_f1 = f1_score(all_start_labels, all_start_preds, zero_division=0)

    end_accuracy = accuracy_score(all_end_labels, all_end_preds)
    end_precision = precision_score(all_end_labels, all_end_preds, zero_division=0)
    end_recall = recall_score(all_end_labels, all_end_preds, zero_division=0)
    end_f1 = f1_score(all_end_labels, all_end_preds, zero_division=0)

    # Compute span-level metrics
    exact_matches = 0
    partial_matches = 0
    total_predictions = 0
    total_ground_truth = 0

    for record in prediction_records:
        pred_spans = eval(record["predicted_spans"])
        gt_spans = eval(record["ground_truth_spans"])

        total_predictions += len(pred_spans)
        total_ground_truth += len(gt_spans)

        for pred_span in pred_spans:
            if pred_span in gt_spans:
                exact_matches += 1
            else:
                # Check for partial overlap
                for gt_span in gt_spans:
                    if (pred_span[0] <= gt_span[1] and pred_span[1] >= gt_span[0]):
                        partial_matches += 1
                        break

    span_precision = exact_matches / total_predictions if total_predictions > 0 else 0
    span_recall = exact_matches / total_ground_truth if total_ground_truth > 0 else 0
    span_f1 = 2 * span_precision * span_recall / (span_precision + span_recall) if (span_precision + span_recall) > 0 else 0

    metrics = {
        f"{split_name}_start_accuracy": start_accuracy,
        f"{split_name}_start_precision": start_precision,
        f"{split_name}_start_recall": start_recall,
        f"{split_name}_start_f1": start_f1,
        f"{split_name}_end_accuracy": end_accuracy,
        f"{split_name}_end_precision": end_precision,
        f"{split_name}_end_recall": end_recall,
        f"{split_name}_end_f1": end_f1,
        f"{split_name}_span_precision": span_precision,
        f"{split_name}_span_recall": span_recall,
        f"{split_name}_span_f1": span_f1,
        f"{split_name}_exact_matches": exact_matches,
        f"{split_name}_partial_matches": partial_matches,
        f"{split_name}_total_predictions": total_predictions,
        f"{split_name}_total_ground_truth": total_ground_truth,
    }

    # Create predictions DataFrame
    predictions_df = pd.DataFrame(prediction_records)

    # Save predictions CSV
    csv_path = output_dir / f"{split_name}_predictions.csv"
    predictions_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {split_name} predictions to {csv_path}")

    return metrics, predictions_df


def main():
    parser = argparse.ArgumentParser(description="Evaluate evidence binding agent")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/train_Evidence",
        help="Directory containing model checkpoint and config"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/train_Evidence/evaluation",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/deberta-v3-base",
        help="Model name for tokenizer"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--span_threshold",
        type=float,
        default=0.5,
        help="Threshold for span prediction"
    )
    parser.add_argument(
        "--posts_path",
        type=str,
        default="Data/ReDSM5/redsm5_posts.csv",
        help="Path to posts CSV"
    )
    parser.add_argument(
        "--annotations_path",
        type=str,
        default="Data/ReDSM5/redsm5_annotations.csv",
        help="Path to annotations CSV"
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint_dir) / "best" / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    # Model configuration
    model_config = {
        "model_name": args.model_name,
        "max_seq_length": args.max_seq_length,
        "dropout": 0.1,
        "label_smoothing": 0.0,
        "max_span_length": 50,
        "span_threshold": args.span_threshold,
    }

    # Load model
    agent, device = load_model(str(checkpoint_path), model_config)

    # Load evidence annotations
    logger.info("Loading evidence annotations...")
    examples = load_evidence_annotations(
        posts_path=args.posts_path,
        annotations_path=args.annotations_path,
        criteria_path=None
    )
    logger.info(f"Loaded {len(examples)} examples")

    # Split data (use same split as training)
    from sklearn.model_selection import train_test_split

    train_examples, temp_examples = train_test_split(
        examples,
        test_size=0.3,
        random_state=42,
        stratify=[ex.has_evidence for ex in examples]
    )

    val_examples, test_examples = train_test_split(
        temp_examples,
        test_size=0.5,
        random_state=42,
        stratify=[ex.has_evidence for ex in temp_examples]
    )

    logger.info(f"Val set size: {len(val_examples)}")
    logger.info(f"Test set size: {len(test_examples)}")

    # Create datasets
    val_dataset = EvidenceDataset(
        examples=val_examples,
        tokenizer_name=args.model_name,
        max_length=args.max_seq_length
    )

    test_dataset = EvidenceDataset(
        examples=test_examples,
        tokenizer_name=args.model_name,
        max_length=args.max_seq_length
    )

    # Create collator
    collator = EvidenceCollator(
        tokenizer_name=args.model_name,
        max_length=args.max_seq_length
    )

    # Create dataloaders
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True
    )

    # Evaluate on validation set
    val_metrics, val_predictions_df = evaluate_dataset(
        agent=agent,
        dataset=val_dataset,
        dataloader=val_loader,
        device=device,
        split_name="val",
        output_dir=output_dir
    )

    # Evaluate on test set
    test_metrics, test_predictions_df = evaluate_dataset(
        agent=agent,
        dataset=test_dataset,
        dataloader=test_loader,
        device=device,
        split_name="test",
        output_dir=output_dir
    )

    # Combine metrics
    all_metrics = {**val_metrics, **test_metrics}

    # Save metrics to JSON
    metrics_path = output_dir / "evaluation_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    print("\nValidation Set Metrics:")
    print(f"  Start Token F1: {val_metrics['val_start_f1']:.4f}")
    print(f"  End Token F1: {val_metrics['val_end_f1']:.4f}")
    print(f"  Span Precision: {val_metrics['val_span_precision']:.4f}")
    print(f"  Span Recall: {val_metrics['val_span_recall']:.4f}")
    print(f"  Span F1: {val_metrics['val_span_f1']:.4f}")
    print(f"  Exact Matches: {val_metrics['val_exact_matches']}/{val_metrics['val_total_ground_truth']}")

    print("\nTest Set Metrics:")
    print(f"  Start Token F1: {test_metrics['test_start_f1']:.4f}")
    print(f"  End Token F1: {test_metrics['test_end_f1']:.4f}")
    print(f"  Span Precision: {test_metrics['test_span_precision']:.4f}")
    print(f"  Span Recall: {test_metrics['test_span_recall']:.4f}")
    print(f"  Span F1: {test_metrics['test_span_f1']:.4f}")
    print(f"  Exact Matches: {test_metrics['test_exact_matches']}/{test_metrics['test_total_ground_truth']}")

    print("\nOutput Files:")
    print(f"  Validation predictions: {output_dir / 'val_predictions.csv'}")
    print(f"  Test predictions: {output_dir / 'test_predictions.csv'}")
    print(f"  Metrics JSON: {metrics_path}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
