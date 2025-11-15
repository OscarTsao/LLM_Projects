"""Training loop utilities."""

from typing import Dict, List, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.losses import (
    adaptive_focal_loss,
    adaptive_focal_loss_multiclass,
    multi_label_loss,
    span_classification_loss,
    token_classification_loss,
)
from src.models.model import EvidenceModel
from src.utils.metrics import (
    compute_multi_label_metrics,
    compute_span_metrics,
    compute_token_metrics,
)


def _resolve_loss_policy(loss_cfg, default_type: str) -> Dict[str, float]:
    """Normalize loss configuration to a consistent policy dictionary."""
    if isinstance(loss_cfg, DictConfig):
        loss_cfg = OmegaConf.to_container(loss_cfg, resolve=True)

    if loss_cfg is None:
        loss_cfg = {}

    if isinstance(loss_cfg, str):
        key = loss_cfg.lower()
        mapping = {
            "bce": "cross_entropy",
            "binary_cross_entropy": "cross_entropy",
            "cross_entropy": "cross_entropy",
            "ce": "cross_entropy",
            "adaptive_focal": "adaptive_focal",
            "focal": "adaptive_focal",
            "hybrid": "hybrid",
        }
        loss_type = mapping.get(key, key)
        return {
            "type": loss_type,
            "cross_entropy_weight": 1.0,
            "adaptive_focal_weight": 1.0,
        }

    if isinstance(loss_cfg, dict):
        loss_type = loss_cfg.get("type", default_type)
        loss_type = loss_type.lower() if isinstance(loss_type, str) else default_type
        return {
            "type": loss_type,
            "cross_entropy_weight": float(loss_cfg.get("cross_entropy_weight", 1.0)),
            "adaptive_focal_weight": float(loss_cfg.get("adaptive_focal_weight", 1.0)),
        }

    if loss_cfg is None:
        return {
            "type": default_type,
            "cross_entropy_weight": 1.0,
            "adaptive_focal_weight": 1.0,
        }

    raise TypeError(f"Unsupported loss configuration type: {type(loss_cfg)}")


def compute_loss(
    head_outputs: Dict[str, Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    head_cfgs: Dict[str, DictConfig],
    loss_weights: DictConfig,
    focal_cfg: DictConfig,
    device: torch.device,
) -> torch.Tensor:
    """Compute loss for all classification heads.

    Args:
        head_outputs: Outputs from all classification heads
        batch: Batch of data
        head_cfgs: Configuration for each head
        loss_weights: Loss weights for each head
        focal_cfg: Focal loss configuration
        device: Device to use for computation

    Returns:
        Combined loss across all heads
    """
    total_loss = 0.0

    for head_name, head_cfg in head_cfgs.items():
        head_type = head_cfg.get("type")
        head_output = head_outputs[head_name]
        loss_policy = _resolve_loss_policy(
            head_cfg.get("loss"),
            default_type="hybrid" if head_type == "multi_label" else "cross_entropy",
        )
        loss_type = loss_policy["type"]

        if head_type == "multi_label":
            targets = batch["multi_labels"].to(device)
            logits = head_output["logits"]

            # Prepare pos_weight
            pos_weight = head_cfg.get("pos_weight")
            pos_weight_tensor = None
            if pos_weight:
                pos_weight_tensor = torch.tensor(pos_weight, device=device, dtype=torch.float32)

            # Compute BCE and focal loss
            label_smoothing = head_cfg.get("label_smoothing", 0.0)
            bce_loss = None
            focal_loss = None

            if loss_type in {"cross_entropy", "hybrid"}:
                bce_loss = multi_label_loss(
                    logits, targets, pos_weight=pos_weight_tensor, label_smoothing=label_smoothing
                )

            if loss_type in {"adaptive_focal", "hybrid"}:
                focal_loss = adaptive_focal_loss(
                    logits,
                    targets,
                    initial_gamma=focal_cfg.initial_gamma,
                    target_positive_rate=focal_cfg.target_positive_rate,
                    alpha=focal_cfg.alpha,
                    min_gamma=focal_cfg.min_gamma,
                    max_gamma=focal_cfg.max_gamma,
                )

            if loss_type == "cross_entropy":
                head_loss = bce_loss
            elif loss_type == "adaptive_focal":
                head_loss = focal_loss
            elif loss_type == "hybrid":
                ce_weight = loss_policy["cross_entropy_weight"]
                focal_weight = loss_policy["adaptive_focal_weight"]
                head_loss = ce_weight * bce_loss + focal_weight * focal_loss
            else:
                raise ValueError(f"Unsupported loss type '{loss_type}' for head '{head_name}'")

            total_loss += loss_weights.get(head_name, 1.0) * head_loss

        elif head_type == "token_classification":
            if "token_labels" not in batch:
                continue

            labels = batch["token_labels"].to(device)
            if (labels != head_cfg.get("ignore_index", -100)).sum() == 0:
                continue

            class_weights = head_cfg.get("class_weights")
            class_weight_tensor = None
            if class_weights:
                class_weight_tensor = torch.tensor(
                    class_weights, device=device, dtype=torch.float32
                )

            logits = head_output["logits"]
            ignore_index = head_cfg.get("ignore_index", -100)

            if loss_type in {"cross_entropy", "hybrid"}:
                ce_loss = token_classification_loss(
                    logits,
                    labels,
                    ignore_index=ignore_index,
                    class_weights=class_weight_tensor,
                )
            else:
                ce_loss = None

            if loss_type in {"adaptive_focal", "hybrid"}:
                focal_loss = adaptive_focal_loss_multiclass(
                    logits,
                    labels,
                    ignore_index=ignore_index,
                    initial_gamma=focal_cfg.initial_gamma,
                    target_positive_rate=focal_cfg.target_positive_rate,
                    alpha=focal_cfg.alpha,
                    min_gamma=focal_cfg.min_gamma,
                    max_gamma=focal_cfg.max_gamma,
                    class_weights=class_weight_tensor,
                )
            else:
                focal_loss = None

            if loss_type == "cross_entropy":
                head_loss = ce_loss
            elif loss_type == "adaptive_focal":
                head_loss = focal_loss
            elif loss_type == "hybrid":
                ce_weight = loss_policy["cross_entropy_weight"]
                focal_weight = loss_policy["adaptive_focal_weight"]
                head_loss = ce_weight * ce_loss + focal_weight * focal_loss
            else:
                raise ValueError(f"Unsupported loss type '{loss_type}' for head '{head_name}'")

            total_loss += loss_weights.get(head_name, 1.0) * head_loss

        elif head_type == "span_classification":
            if "start_positions" not in batch or "end_positions" not in batch:
                continue

            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            if (start_positions != head_cfg.get("ignore_index", -100)).sum() == 0:
                continue

            start_logits = head_output["start_logits"]
            end_logits = head_output["end_logits"]
            start_loss, end_loss = span_classification_loss(
                start_logits,
                end_logits,
                start_positions,
                end_positions,
                ignore_index=head_cfg.get("ignore_index", -100),
            )
            ignore_index = head_cfg.get("ignore_index", -100)

            if loss_type in {"adaptive_focal", "hybrid"}:
                start_focal = adaptive_focal_loss_multiclass(
                    start_logits,
                    start_positions,
                    ignore_index=ignore_index,
                    initial_gamma=focal_cfg.initial_gamma,
                    target_positive_rate=focal_cfg.target_positive_rate,
                    alpha=focal_cfg.alpha,
                    min_gamma=focal_cfg.min_gamma,
                    max_gamma=focal_cfg.max_gamma,
                )
                end_focal = adaptive_focal_loss_multiclass(
                    end_logits,
                    end_positions,
                    ignore_index=ignore_index,
                    initial_gamma=focal_cfg.initial_gamma,
                    target_positive_rate=focal_cfg.target_positive_rate,
                    alpha=focal_cfg.alpha,
                    min_gamma=focal_cfg.min_gamma,
                    max_gamma=focal_cfg.max_gamma,
                )
            else:
                start_focal = None
                end_focal = None

            if loss_type == "cross_entropy":
                start_total = start_loss
                end_total = end_loss
            elif loss_type == "adaptive_focal":
                start_total = start_focal
                end_total = end_focal
            elif loss_type == "hybrid":
                ce_weight = loss_policy["cross_entropy_weight"]
                focal_weight = loss_policy["adaptive_focal_weight"]
                start_total = ce_weight * start_loss + focal_weight * start_focal
                end_total = ce_weight * end_loss + focal_weight * end_focal
            else:
                raise ValueError(f"Unsupported loss type '{loss_type}' for head '{head_name}'")

            total_loss += (
                loss_weights.get(f"{head_name}_start", 1.0) * start_total
                + loss_weights.get(f"{head_name}_end", 1.0) * end_total
            )

    return total_loss


def evaluate(
    model: EvidenceModel,
    dataloader,
    device: torch.device,
    cfg: DictConfig,
    head_thresholds: Dict[str, np.ndarray],
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        device: Device to use for computation
        cfg: Configuration
        head_thresholds: Classification thresholds for each head

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    losses = []
    loss_weights = cfg.training.loss_weights
    focal_cfg = cfg.training.focal
    head_cfgs = model.head_configs

    # Storage for predictions and targets
    multi_label_probs: Dict[str, List[np.ndarray]] = {}
    multi_label_targets: Dict[str, List[np.ndarray]] = {}
    token_logits: Dict[str, List[np.ndarray]] = {}
    token_labels: Dict[str, List[np.ndarray]] = {}
    span_start_logits: Dict[str, List[np.ndarray]] = {}
    span_end_logits: Dict[str, List[np.ndarray]] = {}
    span_start_targets: Dict[str, List[np.ndarray]] = {}
    span_end_targets: Dict[str, List[np.ndarray]] = {}

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Run model inference (autocast disabled for evaluation consistency)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )

            head_outputs = outputs["head_outputs"]
            batch_loss = compute_loss(
                head_outputs, batch, head_cfgs, loss_weights, focal_cfg, device
            )

            # Update progress bar with current loss
            if losses:
                progress_bar.set_postfix({"loss": f"{np.mean(losses):.4f}"})

            # Collect predictions for metrics
            for head_name, head_cfg in head_cfgs.items():
                head_type = head_cfg.get("type")
                head_output = head_outputs[head_name]

                if head_type == "multi_label":
                    targets = batch["multi_labels"].to(device)
                    logits = head_output["logits"]
                    probs = torch.sigmoid(logits).cpu().numpy()
                    multi_label_probs.setdefault(head_name, []).append(probs)
                    multi_label_targets.setdefault(head_name, []).append(targets.cpu().numpy())

                elif head_type == "token_classification":
                    if "token_labels" not in batch:
                        continue
                    labels = batch["token_labels"].to(device)
                    if (labels != head_cfg.get("ignore_index", -100)).sum() == 0:
                        continue
                    logits = head_output["logits"]
                    token_logits.setdefault(head_name, []).append(logits.detach().cpu().numpy())
                    token_labels.setdefault(head_name, []).append(labels.cpu().numpy())

                elif head_type == "span_classification":
                    if "start_positions" not in batch or "end_positions" not in batch:
                        continue
                    start_positions = batch["start_positions"].to(device)
                    end_positions = batch["end_positions"].to(device)
                    if (start_positions != head_cfg.get("ignore_index", -100)).sum() == 0:
                        continue
                    start_logits = head_output["start_logits"]
                    end_logits = head_output["end_logits"]
                    span_start_logits.setdefault(head_name, []).append(
                        start_logits.detach().cpu().numpy()
                    )
                    span_end_logits.setdefault(head_name, []).append(
                        end_logits.detach().cpu().numpy()
                    )
                    span_start_targets.setdefault(head_name, []).append(
                        start_positions.cpu().numpy()
                    )
                    span_end_targets.setdefault(head_name, []).append(end_positions.cpu().numpy())

            losses.append(batch_loss.item() if isinstance(batch_loss, torch.Tensor) else batch_loss)

    # Compute metrics
    avg_loss = float(np.mean(losses)) if losses else 0.0
    metrics: Dict[str, float] = {"val_loss": avg_loss}

    for head_name, probs in multi_label_probs.items():
        y_probs = np.vstack(probs)
        y_true = np.vstack(multi_label_targets[head_name])
        head_metrics = compute_multi_label_metrics(
            head_name, y_true, y_probs, head_thresholds[head_name]
        )
        metrics.update(head_metrics)

    for head_name, logits in token_logits.items():
        head_metrics = compute_token_metrics(
            head_name,
            logits,
            token_labels.get(head_name, []),
            ignore_index=head_cfgs[head_name].get("ignore_index", -100),
        )
        metrics.update(head_metrics)

    for head_name, start_logit_list in span_start_logits.items():
        head_metrics = compute_span_metrics(
            head_name,
            start_logit_list,
            span_end_logits.get(head_name, []),
            span_start_targets.get(head_name, []),
            span_end_targets.get(head_name, []),
            ignore_index=head_cfgs[head_name].get("ignore_index", -100),
        )
        metrics.update(head_metrics)

    return avg_loss, metrics
