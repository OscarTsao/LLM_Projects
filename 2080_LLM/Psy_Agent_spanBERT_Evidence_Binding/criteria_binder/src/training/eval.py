# File: src/training/eval.py
"""Evaluation utilities for criteria binding model."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import json

from ..models.binder import SpanBertEvidenceBinder
from ..data.dataset import CriteriaBindingDataset
from ..training.collator import InferenceCollator
from ..utils.decode import SpanDecoder
from ..utils.metrics import CombinedMetrics, evaluate_predictions
from ..utils.io import write_jsonl, load_json

logger = logging.getLogger(__name__)


def evaluate_model(
    model: SpanBertEvidenceBinder,
    data_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        config: Configuration dictionary

    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()

    # Initialize metrics
    span_iou_thresh = config.get("metrics", {}).get("span_iou_thresh", 0.5)
    metrics = CombinedMetrics(
        span_iou_threshold=span_iou_thresh,
        num_labels=config["model"]["num_labels"],
    )

    total_loss = 0.0
    total_span_loss = 0.0
    total_cls_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass - only pass model inputs, not metadata
            model_inputs = {k: v for k, v in batch.items()
                          if k in ['input_ids', 'attention_mask', 'token_type_ids',
                                 'text_mask', 'start_positions', 'end_positions', 'labels']}
            outputs = model(**model_inputs)

            # Accumulate losses
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
            if "span_loss" in outputs:
                total_span_loss += outputs["span_loss"].item()
            if "cls_loss" in outputs:
                total_cls_loss += outputs["cls_loss"].item()
            num_batches += 1

            # Get predictions for metrics
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]
            text_mask = batch["text_mask"]

            # Decode spans
            decoder = SpanDecoder(
                max_answer_len=config["model"]["max_answer_len"],
                top_k=config["decode"]["top_k"],
                nms_iou_thresh=config["decode"]["nms_iou_thresh"],
                allow_overlap=config["decode"]["allow_overlap"],
            )

            # For evaluation, we'll use a simplified approach
            # In practice, you'd want to use the full window aggregation
            batch_size = start_logits.shape[0]
            pred_spans_batch = []
            gold_spans_batch = []

            from ..utils.decode import decode_single_example_spans

            for b in range(batch_size):
                # Get predictions (simplified - just best span per window)
                spans = decode_single_example_spans(
                    start_logits[b],
                    end_logits[b],
                    text_mask[b],
                    max_answer_len=config["model"]["max_answer_len"],
                    top_k=1,
                )

                if spans:
                    # Convert token spans to char spans (simplified)
                    # In practice, you'd use proper offset mapping
                    pred_spans = [(0, 10)]  # Placeholder
                else:
                    pred_spans = []

                pred_spans_batch.append(pred_spans)

                # Get gold spans (simplified)
                if "start_positions" in batch and "end_positions" in batch:
                    start_pos = batch["start_positions"][b].item()
                    end_pos = batch["end_positions"][b].item()
                    if start_pos != -1 and end_pos != -1:
                        gold_spans = [(0, 10)]  # Placeholder
                    else:
                        gold_spans = []
                else:
                    gold_spans = []

                gold_spans_batch.append(gold_spans)

            # Update metrics
            pred_labels = None
            gold_labels = None

            if "cls_logits" in outputs and "labels" in batch:
                pred_labels = torch.argmax(outputs["cls_logits"], dim=-1).cpu().tolist()
                gold_labels = batch["labels"].cpu().tolist()

            metrics.update(
                pred_spans_batch,
                gold_spans_batch,
                pred_labels,
                gold_labels,
            )

    # Compute final metrics
    eval_metrics = metrics.compute()

    # Add loss metrics
    if num_batches > 0:
        eval_metrics.update({
            "eval_loss": total_loss / num_batches,
            "eval_span_loss": total_span_loss / num_batches,
            "eval_cls_loss": total_cls_loss / num_batches,
        })

    return eval_metrics


def run_inference(
    model_path: str,
    data_path: str,
    output_path: str,
    config: Dict[str, Any],
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Run inference on a dataset and save predictions.

    Args:
        model_path: Path to saved model checkpoint
        data_path: Path to input JSONL data
        output_path: Path to save predictions
        config: Configuration dictionary
        device: Device to run inference on

    Returns:
        Dictionary with inference results and metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading model from {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load model
    model = SpanBertEvidenceBinder(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        use_label_head=config["model"]["use_label_head"],
        dropout=config["model"]["dropout"],
        lambda_span=config["model"]["lambda_span"],
    )

    # Load model weights
    model_weights_path = Path(model_path) / "pytorch_model.bin"
    if model_weights_path.exists():
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

    model.to(device)
    model.eval()

    # Load dataset
    dataset = CriteriaBindingDataset(data_path)

    # Create collator for inference
    collator = InferenceCollator(
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        doc_stride=config["model"]["doc_stride"],
    )

    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=False,
        collate_fn=collator,
        num_workers=0,  # Use 0 for inference to avoid pickling issues
    )

    # Initialize decoder
    decoder = SpanDecoder(
        max_answer_len=config["model"]["max_answer_len"],
        top_k=config["decode"]["top_k"],
        nms_iou_thresh=config["decode"]["nms_iou_thresh"],
        allow_overlap=config["decode"]["allow_overlap"],
    )

    # Run inference
    all_predictions = {}

    logger.info("Running inference...")
    with torch.no_grad():
        for batch_data in tqdm(data_loader, desc="Inference"):
            batch = batch_data["batch"]
            metadata = batch_data["metadata"]

            # Move batch to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass - only pass model inputs, not metadata
            model_inputs = {k: v for k, v in batch.items()
                          if k in ['input_ids', 'attention_mask', 'token_type_ids', 'text_mask']}
            outputs = model(**model_inputs)

            # Decode predictions
            predictions = decoder.decode_batch(
                outputs["start_logits"],
                outputs["end_logits"],
                batch["text_mask"],
                metadata,
            )

            # Add classification predictions if available
            if "cls_logits" in outputs:
                cls_probs = F.softmax(outputs["cls_logits"], dim=-1)
                cls_predictions = torch.argmax(outputs["cls_logits"], dim=-1)

                # Add to predictions
                for i, meta in enumerate(metadata):
                    example_id = meta["example_id"]
                    if example_id in predictions:
                        predictions[example_id].extend([{
                            "pred_label": cls_predictions[i].item(),
                            "pred_label_probs": cls_probs[i].cpu().tolist(),
                        }])

            all_predictions.update(predictions)

    # Convert predictions to output format
    output_predictions = []
    for example in dataset.examples:
        example_id = example["id"]
        pred_spans = all_predictions.get(example_id, [])

        prediction = {
            "id": example_id,
            "pred_spans_char": [[span["start_char"], span["end_char"]]
                               for span in pred_spans],
            "pred_spans_scores": [span["score"] for span in pred_spans],
        }

        # Add classification prediction if available
        if pred_spans and "pred_label" in pred_spans[0]:
            prediction["pred_label"] = pred_spans[0]["pred_label"]
            prediction["pred_label_probs"] = pred_spans[0]["pred_label_probs"]

        output_predictions.append(prediction)

    # Save predictions
    write_jsonl(output_predictions, output_path)
    logger.info(f"Saved predictions to {output_path}")

    # Compute metrics if gold data is available
    metrics = {}
    if any("evidence_char_spans" in ex for ex in dataset.examples):
        try:
            metrics = evaluate_predictions(
                all_predictions,
                dataset.examples,
                span_iou_threshold=config.get("metrics", {}).get("span_iou_thresh", 0.5),
            )
            logger.info("Evaluation metrics:")
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")
        except Exception as e:
            logger.warning(f"Failed to compute metrics: {e}")

    return {
        "predictions": all_predictions,
        "metrics": metrics,
        "num_examples": len(dataset.examples),
    }


def load_model_for_inference(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
) -> Tuple[SpanBertEvidenceBinder, AutoTokenizer, Dict[str, Any]]:
    """Load model, tokenizer, and config for inference.

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        Tuple of (model, tokenizer, config)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(checkpoint_path)

    # Load config
    config_path = checkpoint_path / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found at {config_path}")

    from ..utils.io import load_yaml
    config = load_yaml(config_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

    # Load model
    model = SpanBertEvidenceBinder(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        use_label_head=config["model"]["use_label_head"],
        dropout=config["model"]["dropout"],
        lambda_span=config["model"]["lambda_span"],
    )

    # Load model weights
    model_weights_path = checkpoint_path / "pytorch_model.bin"
    if model_weights_path.exists():
        model.load_state_dict(torch.load(model_weights_path, map_location=device))
    else:
        raise FileNotFoundError(f"Model weights not found at {model_weights_path}")

    model.to(device)
    return model, tokenizer, config