from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple
import os

try:
    import optuna  # type: ignore
except Exception:  # pragma: no cover - optional
    optuna = None  # type: ignore

from src.utils.mlflow_buffer import MlflowBufferedLogger
from src.utils.oom_guard import catch_oom, maybe_prune_for_oom, OOMDuringTraining
from src.utils.data import load_dataset


def _try_import_transformers():
    try:  # defer heavy imports
        import torch  # type: ignore
        from torch import nn  # type: ignore
        from torch.utils.data import Dataset, DataLoader  # type: ignore
        from transformers import AutoModel, AutoTokenizer, get_scheduler  # type: ignore

        return torch, nn, Dataset, DataLoader, AutoModel, AutoTokenizer, get_scheduler
    except Exception:
        return None, None, None, None, None, None, None


def _should_use_hf_backend() -> bool:
    # Enable with env flag or cfg override
    return os.environ.get("USE_HF_TRAIN", "0") in ("1", "true", "yes")


@dataclass
class TrainOutputs:
    val_metric: float
    epochs: int
    duration_seconds: float


def _pseudo_val_metric(hparams: Dict[str, Any], epoch: int, base: float = 0.6) -> float:
    """Deterministic surrogate objective that changes with hparams and epoch.

    This avoids heavy DL dependencies but enables Optuna pruning/optimization.
    """
    lr = float(hparams.get("learning_rate", 3e-5))
    wd = float(hparams.get("weight_decay", 0.01))
    dr = float(hparams.get("dropout", 0.1))
    warm = float(hparams.get("warmup_ratio", 0.05))
    # Normalize ranges
    s_lr = max(0.0, min(1.0, (math.log10(lr + 1e-12) + 6) / 3.0))
    s_wd = max(0.0, min(1.0, (math.log10(wd + 1e-12) + 6) / 6.0))
    s_dr = max(0.0, min(1.0, (0.8 - dr) / 0.8))
    s_wm = max(0.0, min(1.0, (0.2 - abs(warm - 0.05)) / 0.2))
    prog = 1.0 - math.exp(-0.6 * (epoch + 1))
    score = base + 0.2 * s_lr + 0.1 * s_wd + 0.15 * s_dr + 0.15 * s_wm
    score = score * (0.8 + 0.2 * prog)
    return float(max(0.0, min(0.999, score)))


def train_model(
    cfg: Dict[str, Any],
    *,
    trial=None,
    epochs: int = 5,
    logger: Optional[MlflowBufferedLogger] = None,
) -> TrainOutputs:
    """Minimal training loop with epoch-wise reporting and pruning.

    - Validates batch_size x max_length for OOM risk.
    - Reports `val_metric` to Optuna per epoch.
    - Logs metrics to Mlflow via buffered logger.
    - Raises optuna.TrialPruned on prune signal or OOM.
    """
    # Try HF backend unless disabled or unavailable
    if _should_use_hf_backend():
        torch, nn, Dataset, DataLoader, AutoModel, AutoTokenizer, get_scheduler = _try_import_transformers()
        if torch is not None:
            return _train_model_hf(cfg, trial=trial, epochs=epochs, logger=logger, torch=torch, nn=nn, Dataset=Dataset, DataLoader=DataLoader, AutoModel=AutoModel, AutoTokenizer=AutoTokenizer, get_scheduler=get_scheduler)

    t0 = time.time()
    params = cfg.get("params", {})
    batch = int(params.get("batch_size", 8))
    max_len = int(params.get("max_length", 256))
    maybe_prune_for_oom(batch, max_len)

    val_metric = 0.0
    with catch_oom():
        for epoch in range(epochs):
            # ... training would happen here ...
            # Compute val metric surrogate
            val_metric = _pseudo_val_metric(params, epoch)

            if logger is not None:
                logger.log_metrics({"val_metric": val_metric, "epoch": epoch + 1}, step=epoch + 1)

            if trial is not None and optuna is not None:
                trial.report(val_metric, step=epoch + 1)
                if trial.should_prune():
                    if logger is not None:
                        logger.set_tags({"pruned": "true", "pruned_epoch": str(epoch + 1)})
                    raise optuna.TrialPruned(f"Pruned at epoch {epoch + 1}")

    duration = time.time() - t0
    if logger is not None:
        logger.log_metrics({"duration_seconds": duration})
    return TrainOutputs(val_metric=val_metric, epochs=epochs, duration_seconds=duration)


def _build_examples_from_dataset(dataset: List[Dict[str, Any]]) -> List[Tuple[str, str, int]]:
    """Create (symptom, sentence_text, label) examples from dataset."""
    ex: List[Tuple[str, str, int]] = []
    for item in dataset:
        sentences = {str(s.get("sentence_id")): s.get("text", "") for s in item.get("sentences", [])}
        for lab in item.get("labels", []):
            sid = str(lab.get("sentence_id"))
            symptom = str(lab.get("symptom", ""))
            text = sentences.get(sid, "")
            y = int(lab.get("status", 0))
            if text:
                ex.append((symptom, text, y))
    return ex


def _mk_dataloaders(
    torch,
    Dataset,
    DataLoader,
    tokenizer,
    examples: List[Tuple[str, str, int]],
    batch_size: int,
    max_length: int,
    split_ratio: float = 0.2,
):
    class PairDataset(Dataset):  # type: ignore[misc, valid-type]
        def __init__(self, ex):
            self.ex = ex

        def __len__(self):  # noqa: D401
            return len(self.ex)

        def __getitem__(self, idx):  # noqa: D401
            symptom, sentence, y = self.ex[idx]
            text = f"{symptom} [SEP] {sentence}"
            enc = tokenizer(text, truncation=True, max_length=max_length, padding=False)
            return {
                "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "labels": torch.tensor(y, dtype=torch.long),
            }

    n = len(examples)
    n_val = max(1, int(n * split_ratio))
    train_ex = examples[:-n_val] if n_val < n else examples
    val_ex = examples[-n_val:] if n_val < n else examples

    def collate(features):
        ids = [f["input_ids"] for f in features]
        att = [f["attention_mask"] for f in features]
        labels = torch.stack([f["labels"] for f in features])
        pad_id = tokenizer.pad_token_id or 0
        input_ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
        attention_mask = torch.nn.utils.rnn.pad_sequence(att, batch_first=True, padding_value=0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Hardware-tuned DataLoaders
    env_workers = os.environ.get("DL_NUM_WORKERS")
    if env_workers is not None:
        try:
            num_workers = int(env_workers)
        except Exception:
            num_workers = 2
    else:
        try:
            cpu_ct = os.cpu_count() or 4
            num_workers = max(2, min(8, cpu_ct // 2))
        except Exception:
            num_workers = 2
    pin_memory = True
    persistent = num_workers > 0
    prefetch = int(os.environ.get("DL_PREFETCH", "4")) if num_workers > 0 else None

    dl_train = DataLoader(
        PairDataset(train_ex),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch if prefetch is not None else 2,
    )
    dl_val = DataLoader(
        PairDataset(val_ex),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        prefetch_factor=prefetch if prefetch is not None else 2,
    )
    return dl_train, dl_val


def _build_head(nn, hidden_size: int, head: str, dropout: float, num_labels: int = 2):
    if head == "mlp":
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
    # For attn pooling, the head remains linear; pooling handled before
    return nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(hidden_size, num_labels),
    )


def _pool_outputs(nn, outputs, attention_mask, pooling: str, dropout: float):
    last_hidden = outputs.last_hidden_state  # [B, T, H]
    if pooling == "mean":
        mask = attention_mask.unsqueeze(-1).float()
        summed = (last_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return summed / denom
    # default cls
    return last_hidden[:, 0]


def _compute_val_metric(torch, logits, labels) -> float:
    preds = torch.argmax(logits, dim=-1)
    y = labels
    # macro F1 for positive class (present)
    tp = int(((preds == 1) & (y == 1)).sum().item())
    fp = int(((preds == 1) & (y == 0)).sum().item())
    fn = int(((preds == 0) & (y == 1)).sum().item())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return float(f1)


def _train_model_hf(
    cfg: Dict[str, Any],
    *,
    trial=None,
    epochs: int,
    logger: Optional[MlflowBufferedLogger],
    torch,
    nn,
    Dataset,
    DataLoader,
    AutoModel,
    AutoTokenizer,
    get_scheduler,
) -> TrainOutputs:
    params = cfg.get("params", {})
    data_cfg = cfg.get("data", {})
    data_path = data_cfg.get("path") or "data/redsm5_sample.jsonl"
    examples = _build_examples_from_dataset(list(load_dataset(data_path)))

    batch = int(params.get("batch_size", 8))
    max_len = int(params.get("max_length", 256))
    maybe_prune_for_oom(batch, max_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Enable TF32 where supported for speed
    try:
        if device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
            try:
                torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
            except Exception:
                pass
    except Exception:
        pass
    model_name = cfg.get("model", {}).get("name", "microsoft/deberta-v3-base")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dl_train, dl_val = _mk_dataloaders(torch, Dataset, DataLoader, tokenizer, examples, batch, max_len)

    base_model = AutoModel.from_pretrained(model_name)
    # Optional gradient checkpointing for stability (env: GRAD_CKPT=1)
    if os.environ.get("GRAD_CKPT", "0") in ("1", "true", "yes"):
        try:
            base_model.gradient_checkpointing_enable()
        except Exception:
            pass
    hidden = int(getattr(base_model.config, "hidden_size", 768))

    head_type = str(params.get("head", "linear"))
    pooling = str(params.get("pooling", "cls"))
    dropout = float(params.get("dropout", 0.1))
    classifier = _build_head(nn, hidden, head=head_type, dropout=dropout, num_labels=2)

    class PairClassifier(nn.Module):  # type: ignore[misc, valid-type]
        def __init__(self, base, clf, pooling):
            super().__init__()
            self.base = base
            self.classifier = clf
            self.pooling = pooling
            self.dropout = nn.Dropout(dropout)

        def forward(self, input_ids, attention_mask):  # noqa: D401
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            pooled = _pool_outputs(nn, out, attention_mask, pooling=self.pooling, dropout=dropout)
            logits = self.classifier(pooled)
            return logits

    model = PairClassifier(base_model, classifier, pooling).to(device)

    # Optimizer
    lr = float(params.get("learning_rate", 3e-5))
    wd = float(params.get("weight_decay", 0.01))
    opt_name = str(params.get("optimizer", "adamw"))
    if opt_name == "lion":
        try:
            from lion_pytorch import Lion  # type: ignore

            optimizer = Lion(model.parameters(), lr=lr, weight_decay=wd)
        except Exception:
            try:
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=(device.type == "cuda"))
            except TypeError:
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        try:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=(device.type == "cuda"))
        except TypeError:
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    num_update_steps_per_epoch = max(1, len(dl_train))
    num_training_steps = epochs * num_update_steps_per_epoch
    sched_name = str(params.get("scheduler", "linear"))
    if sched_name == "one_cycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=num_training_steps)
        scheduler_step_on_batch = True
    else:
        scheduler = get_scheduler(
            sched_name if sched_name in ("linear", "cosine") else "linear",
            optimizer=optimizer,
            num_warmup_steps=int(num_training_steps * float(params.get("warmup_ratio", 0.05))),
            num_training_steps=num_training_steps,
        )
        scheduler_step_on_batch = False

    # Loss
    loss_name = str(params.get("loss", "ce")).lower()
    if loss_name == "focal":
        def focal_loss(logits, labels, gamma: float = 2.0):
            ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
            pt = torch.exp(-ce)
            return ((1 - pt) ** gamma * ce).mean()

        criterion = focal_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()

    # Optional compile for speed (PyTorch 2; env: USE_TORCH_COMPILE=1)
    use_compile = os.environ.get("USE_TORCH_COMPILE", "1") in ("1", "true", "yes")
    if use_compile:
        try:
            model = torch.compile(model, mode=os.environ.get("TORCH_COMPILE_MODE", "default"))  # type: ignore[attr-defined]
        except Exception:
            pass

    # AMP setup
    amp_dtype = torch.bfloat16 if (device.type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()) else torch.float16  # type: ignore[attr-defined]
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))  # type: ignore[attr-defined]

    # Train loop
    t0 = time.time()
    best_f1 = 0.0
    with catch_oom():
        for epoch in range(epochs):
            model.train()
            for batch in dl_train:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):  # type: ignore[attr-defined]
                    logits = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(logits, labels)
                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                if scheduler_step_on_batch:
                    scheduler.step()
            if not scheduler_step_on_batch:
                scheduler.step()

            # Validation
            model.eval()
            total_logits = []
            total_labels = []
            with torch.no_grad():
                for batch in dl_val:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)
                    with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=amp_dtype):  # type: ignore[attr-defined]
                        logits = model(input_ids=input_ids, attention_mask=attention_mask)
                total_logits.append(logits.cpu())
                total_labels.append(labels.cpu())
            logits_cat = torch.cat(total_logits) if total_logits else torch.empty((0, 2))
            labels_cat = torch.cat(total_labels) if total_labels else torch.empty((0,), dtype=torch.long)
            val_f1 = _compute_val_metric(torch, logits_cat, labels_cat)

            if logger is not None:
                logger.log_metrics({"val_metric": val_f1, "epoch": epoch + 1}, step=epoch + 1)
            if trial is not None:
                trial.report(val_f1, step=epoch + 1)
                if trial.should_prune():
                    if logger is not None:
                        logger.set_tags({"pruned": "true", "pruned_epoch": str(epoch + 1)})
                    raise __import__("optuna").TrialPruned(f"Pruned at epoch {epoch + 1}")
            best_f1 = max(best_f1, val_f1)

    duration = time.time() - t0
    if logger is not None:
        logger.log_metrics({"duration_seconds": duration})
    return TrainOutputs(val_metric=best_f1, epochs=epochs, duration_seconds=duration)
