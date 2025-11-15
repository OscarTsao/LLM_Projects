from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

import optuna
import torch
from jsonschema import ValidationError, validate
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataaug_multi_both.data import load_raw_datasets, tokenize_datasets
from dataaug_multi_both.model.multitask import build_multitask_model
from dataaug_multi_both.training.collator import DynamicPaddingCollator
from dataaug_multi_both.training.metrics import compute_metrics


def _build_dataloader(dataset, cfg, split: str) -> DataLoader:
    tokenizer_name = cfg["encoder"]["tokenizer_name"]
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    collator = DynamicPaddingCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=int(cfg["tokenizer"]["max_length"]),
    )
    return DataLoader(dataset[split], batch_size=cfg["train"]["per_device_batch_size"], collate_fn=collator)


@torch.no_grad()
def evaluate_checkpoint(
    cfg: Mapping[str, Any],
    checkpoint_path: Path,
    split: str = "validation",
    output_path: Optional[Path] = None,
) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(cfg["encoder"]["tokenizer_name"])
    raw_dataset, metadata = load_raw_datasets(cfg)
    tokenized, _ = tokenize_datasets(raw_dataset, cfg, tokenizer)

    dataloader = _build_dataloader(tokenized, cfg, split)

    model = build_multitask_model(cfg)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logits_ev, logits_cr, labels_ev, labels_cr = [], [], [], []
    for batch in dataloader:
        batch = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        outputs = model(batch["input_ids"], batch["attention_mask"])
        logits_ev.append(outputs["evidence_logits"].cpu())
        logits_cr.append(outputs["criteria_logits"].cpu())
        labels_ev.append(batch["evidence_label"].cpu())
        labels_cr.append(batch["criteria_label"].cpu())

    logits_ev = torch.cat(logits_ev).numpy()
    logits_cr = torch.cat(logits_cr).numpy()
    labels_ev = torch.cat(labels_ev).numpy()
    labels_cr = torch.cat(labels_cr).numpy()

    metrics = compute_metrics(logits_ev, logits_cr, labels_ev, labels_cr, split=split)
    report = {
        "split": split,
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path),
    }

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report


def summarize_study_results(
    study_name: str,
    storage: str,
    schema_path: Path,
    output_path: Path,
    top_k: int = 3,
) -> None:
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [trial for trial in study.get_trials(deepcopy=False) if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        raise RuntimeError("No completed trials found for study summary")

    best_trial = study.best_trial
    summary = {
        "report_id": str(uuid.uuid4()),
        "study_id": str(uuid.uuid5(uuid.NAMESPACE_URL, study_name)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_trial_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{study_name}-{best_trial.number}")),
        "best_validation_score": best_trial.value,
        "best_trial_report_path": best_trial.user_attrs.get("evaluation_report_path", ""),
        "optimization_metric_name": study.direction.name,
        "trials_count": len(completed),
        "top_trials": [
            {
                "trial_id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{study_name}-{trial.number}")),
                "validation_score": trial.value,
            }
            for trial in sorted(completed, key=lambda t: t.value, reverse=True)[:top_k]
        ],
        "report_file_path": str(output_path),
    }

    schema = json.loads(schema_path.read_text())
    try:
        validate(instance=summary, schema=schema)
    except ValidationError as exc:
        raise SystemExit(f"Study summary failed validation: {exc.message}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
