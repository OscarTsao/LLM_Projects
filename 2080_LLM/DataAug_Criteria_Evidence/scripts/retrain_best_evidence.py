#!/usr/bin/env python3
"""
Retrain the best Evidence model from an Optuna study and persist artifacts.

Artifacts saved per run:
- config.json: fully decoded training configuration
- checkpoint.pt: model state_dict (best by validation metric)
- metrics.json: validation + test metrics and runtime

Usage examples:
  # Retrain best evidence model (with augmentation settings from the study)
  python scripts/retrain_best_evidence.py \
    --study aug-evidence-production-2025-10-27 \
    --storage sqlite:///_optuna/dataaug.db \
    --outdir outputs/retrain/evidence_aug

  # Retrain same model but force augmentation off (original dataset)
  python scripts/retrain_best_evidence.py \
    --study aug-evidence-production-2025-10-27 \
    --storage sqlite:///_optuna/dataaug.db \
    --outdir outputs/retrain/evidence_noaug \
    --disable-augmentation
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer


def _decode_params(cur: sqlite3.Cursor, trial_id: int) -> dict[str, Any]:
    rows = cur.execute(
        "select param_name, param_value, distribution_json from trial_params where trial_id=?",
        (trial_id,),
    ).fetchall()
    params: dict[str, Any] = {}
    for name, val, dist_json in rows:
        try:
            dist = json.loads(dist_json) if dist_json else None
        except Exception:
            dist = None
        if dist and dist.get("name") == "CategoricalDistribution":
            choices = dist.get("attributes", {}).get("choices", [])
            try:
                idx = int(val)
                params[name] = choices[idx] if 0 <= idx < len(choices) else val
            except Exception:
                params[name] = val
        else:
            params[name] = val
    return params


def _extract_methods(params: dict[str, Any]) -> list[str]:
    methods: list[str] = []

    raw_methods = params.get("aug.methods")
    if isinstance(raw_methods, str):
        methods.extend(m.strip() for m in raw_methods.split(";") if m.strip())
    elif isinstance(raw_methods, (list, tuple)):
        methods.extend(str(m) for m in raw_methods if m)

    legacy_keys = (
        "aug.nlpaug_method_1",
        "aug.nlpaug_method_2",
        "aug.nlpaug_method_3",
        "aug.textattack_method_1",
        "aug.textattack_method_2",
    )
    for key in legacy_keys:
        value = params.get(key)
        if value:
            methods.append(str(value))

    for key, value in params.items():
        if key.startswith("aug.method[") and value:
            name = key[len("aug.method[") : -1]
            methods.append(str(name))
        if key.startswith("aug.active[") and value:
            name = key[len("aug.active[") : -1]
            methods.append(str(name))

    seen: set[str] = set()
    unique: list[str] = []
    for method in methods:
        if method in seen or not method:
            continue
        seen.add(method)
        unique.append(method)
    return unique


def _extract_method_weights(params: dict[str, Any]) -> dict[str, float] | None:
    weights: dict[str, float] = {}
    for key, value in params.items():
        if key.startswith("aug.weight["):
            name = key[len("aug.weight[") : -1]
            try:
                weights[name] = float(value)
            except (TypeError, ValueError):
                continue
    logits: dict[str, float] = {}
    for key, value in params.items():
        if key.startswith("aug.logit["):
            name = key[len("aug.logit[") : -1]
            try:
                logits[name] = float(value)
            except (TypeError, ValueError):
                continue
    if logits:
        max_logit = max(logits.values())
        exp_vals = {name: math.exp(val - max_logit) for name, val in logits.items()}
        total = sum(exp_vals.values()) or 1.0
        for name, val in exp_vals.items():
            weights[name] = val / total
    return weights or None


def _extract_method_kwargs(params: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    method_kwargs: dict[str, dict[str, Any]] = {}

    if "aug.random_char.action" in params:
        method_kwargs.setdefault("nlpaug/char/RandomCharAug", {})["action"] = params[
            "aug.random_char.action"
        ]

    if "aug.random_word.action" in params:
        method_kwargs.setdefault("nlpaug/word/RandomWordAug", {})["action"] = params[
            "aug.random_word.action"
        ]

    if "aug.tfidf.action" in params:
        method_kwargs.setdefault("nlpaug/word/TfIdfAug", {})["action"] = params[
            "aug.tfidf.action"
        ]
    if "aug.tfidf.top_k" in params:
        method_kwargs.setdefault("nlpaug/word/TfIdfAug", {})["top_k"] = params[
            "aug.tfidf.top_k"
        ]

    return method_kwargs or None


def _build_cfg_from_params(params: dict[str, Any]) -> dict[str, Any]:
    # Mirror scripts/tune_max.py build_config structure for evidence
    cfg: dict[str, Any] = {
        "task": "evidence",
        "model": {"name": params.get("model.name", "roberta-base")},
        "tok": {
            "max_length": int(params.get("tok.max_length", 160)),
            "doc_stride": int(params.get("tok.doc_stride", 32)),
            "use_fast": bool(params.get("tok.use_fast", True)),
        },
        "optim": {
            "name": params.get("optim.name", "adamw"),
            "lr": float(params.get("optim.lr", 3e-5)),
            "weight_decay": float(params.get("optim.weight_decay", 1e-5)),
            "beta1": params.get("optim.beta1"),
            "beta2": params.get("optim.beta2"),
            "eps": params.get("optim.eps"),
            "layerwise_lr_decay": params.get("optim.layerwise_lr_decay"),
        },
        "sched": {
            "name": params.get("sched.name", "linear"),
            "warmup_ratio": float(params.get("sched.warmup_ratio", 0.0)),
        },
        "regularization": {
            "dropout": float(params.get("model.dropout", 0.1)),
            "attn_dropout": float(params.get("model.attn_dropout", 0.0)),
        },
        "head": {
            "layers": int(params.get("head.layers", 1)),
            "hidden": params.get("head.hidden", 384),
            "activation": params.get("head.activation", "gelu"),
            "dropout": float(params.get("head.dropout", 0.1)),
        },
        "loss": {
            "type": params.get("loss.qa.type", "qa_ce"),
            "label_smoothing": float(params.get("loss.qa.label_smoothing", 0.0)),
        },
        "qa": {
            "null": {
                "policy": params.get("qa.null.policy", "none"),
                "threshold": params.get("qa.null.threshold"),
                "ratio": params.get("qa.null.ratio"),
            },
            "topk": int(params.get("qa.topk", 1)),
            "max_answer_len": int(params.get("qa.max_answer_len", 128)),
            "n_best_size": int(params.get("qa.n_best_size", 20)),
            "reranker": params.get("qa.reranker", "sum"),
            "nms_iou": float(params.get("qa.nms_iou", 0.5)),
            "neg_ratio": float(params.get("qa.neg_ratio", 0.5)),
        },
        "train": {
            "batch_size": int(params.get("train.batch_size", 16)),
            "grad_accum": int(params.get("train.grad_accum", 1)),
            "epochs": int(os.getenv("RETRAIN_EPOCHS", os.getenv("HPO_EPOCHS", "12"))),
            "clip_grad": float(params.get("train.clip_grad", 1.0)),
            "grad_checkpointing": bool(params.get("train.grad_checkpointing", False)),
            "freeze_encoder_layers": int(params.get("train.freeze_encoder_layers", 0)),
        },
        "augmentation": {
            "enabled": bool(params.get("aug.enabled", False)),
            "methods": _extract_methods(params),
            "p_apply": float(params.get("aug.p_apply", 0.15)),
            "ops_per_sample": int(params.get("aug.ops_per_sample", 1)),
            "max_replace": float(
                params.get("aug.max_replace", params.get("aug.max_replace_ratio", 0.3))
            ),
            "method_weights": _extract_method_weights(params),
            "method_kwargs": _extract_method_kwargs(params),
            "tfidf_model_path": params.get("aug.tfidf_model_path"),
            "reserved_map_path": params.get("aug.reserved_map_path"),
            "seed": int(params.get("seed", 42)),
        },
        "meta": {"seed": int(params.get("seed", 42))},
    }
    return cfg


def _flatten(d: dict[str, Any], parent: str = "", sep: str = ".") -> dict[str, Any]:
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else k
        if isinstance(v, dict):
            items.extend(_flatten(v, key, sep).items())
        else:
            items.append((key, v))
    return dict(items)


def _setup_mlflow(experiment: str = "evidence-SINGLE") -> None:
    """Configure MLflow to use local sqlite backend and file artifacts.

    Tracking URI can be overridden via MLFLOW_TRACKING_URI; otherwise defaults
    to sqlite:///mlflow.db in the project root. Artifacts are stored under ./mlruns.
    """
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI", f"sqlite:///{Path('mlflow.db').absolute()}"
    )
    mlflow.set_tracking_uri(tracking_uri)
    # Ensure experiment exists with explicit artifact root under ./mlruns
    art_root = Path("mlruns").absolute().as_uri()
    try:
        exp_id = mlflow.create_experiment(experiment, artifact_location=art_root)
    except Exception:
        exp = mlflow.get_experiment_by_name(experiment)
        if exp is None:
            raise
        exp_id = exp.experiment_id
    mlflow.set_experiment(experiment)


def train_and_eval(cfg: dict[str, Any], outdir: Path) -> dict[str, Any]:
    """Train Evidence model and save artifacts; return metrics dict."""
    outdir.mkdir(parents=True, exist_ok=True)

    # Save config
    (outdir / "config.json").write_text(json.dumps(cfg, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg["model"]["name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Import Project.Evidence classes (same as HPO bridge)
    from Project.Evidence.data.dataset import EvidenceDataset
    from Project.Evidence.models.model import Model as EvidenceModel

    # Build full dataset twice (train vs eval) to avoid leaking augmentation (not used for QA anyway)
    data_csv = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "redsm5_matched_evidence.csv"
    )
    train_full = EvidenceDataset(
        csv_path=data_csv,
        tokenizer=tokenizer,
        max_length=cfg["tok"]["max_length"],
        augmentation_pipeline=None,
        is_training=True,
    )
    eval_full = EvidenceDataset(
        csv_path=data_csv,
        tokenizer=tokenizer,
        max_length=cfg["tok"]["max_length"],
        augmentation_pipeline=None,
        is_training=False,
    )

    # Deterministic split (80/10/10)
    total = len(train_full)
    tr = int(0.8 * total)
    va = int(0.1 * total)
    te = total - tr - va
    gen = torch.Generator().manual_seed(cfg["meta"]["seed"])
    idx = torch.randperm(total, generator=gen).tolist()
    idx_tr, idx_va, idx_te = idx[:tr], idx[tr : tr + va], idx[tr + va :]

    ds_tr = Subset(train_full, idx_tr)
    ds_va = Subset(eval_full, idx_va)
    ds_te = Subset(eval_full, idx_te)

    nw = int(os.getenv("NUM_WORKERS", "8"))
    bs = int(cfg["train"]["batch_size"])
    train_loader = DataLoader(
        ds_tr,
        batch_size=bs,
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        ds_va,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_te,
        batch_size=bs,
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
    )

    # Model + optim
    model = EvidenceModel(model_name=model_name, head_cfg=cfg.get("head", {})).to(
        device
    )
    criterion = nn.CrossEntropyLoss()
    optim_name = str(cfg["optim"]["name"]).lower()
    lr = float(cfg["optim"]["lr"])
    wd = (
        float(cfg["optim"]["weight_decay"])
        if cfg["optim"].get("weight_decay") is not None
        else 0.0
    )
    if optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Training loop with best checkpoint in-memory
    use_amp = torch.cuda.is_available()
    dtype = (
        torch.bfloat16
        if (use_amp and torch.cuda.get_device_capability()[0] >= 8)
        else torch.float16
    )
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_em = -1.0
    best_precision = 0.0
    best_recall = 0.0
    best_f1 = 0.0
    best_state = None
    start = time.time()
    # EarlyStopping with patience (default 20, configurable via RETRAIN_PATIENCE)
    patience = int(os.getenv("RETRAIN_PATIENCE", "20"))
    bad_epochs = 0

    # Start MLflow run for single-train with timestamped name
    ts = time.strftime("%Y%m%d-%H%M%S")
    mlflow.start_run(run_name=f"single-train-{ts}")
    # Log flattened config as params
    try:
        mlflow.log_params(
            {
                k: v
                for k, v in _flatten(cfg).items()
                if isinstance(v, (str, int, float, bool)) or v is None
            }
        )
    except Exception:
        pass

    def _accumulate_span_metrics(
        sp: torch.Tensor,
        ep: torch.Tensor,
        start_pos: torch.Tensor,
        end_pos: torch.Tensor,
    ) -> tuple[float, float, float, float]:
        """Return per-batch (sum) metrics: exact match, precision, recall, f1."""
        # Ensure tensors are on CPU for safe scalar math
        sp_cpu = sp.detach().cpu()
        ep_cpu = ep.detach().cpu()
        start_cpu = start_pos.detach().cpu()
        end_cpu = end_pos.detach().cpu()
        em_sum = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        for ps, pe, gs, ge in zip(
            sp_cpu.tolist(),
            ep_cpu.tolist(),
            start_cpu.tolist(),
            end_cpu.tolist(),
            strict=False,
        ):
            # Normalize spans (start <= end)
            p_start, p_end = (ps, pe) if ps <= pe else (pe, ps)
            g_start, g_end = (gs, ge) if gs <= ge else (ge, gs)
            pred_len = max(0, p_end - p_start + 1)
            gold_len = max(0, g_end - g_start + 1)
            overlap = max(0, min(p_end, g_end) - max(p_start, g_start) + 1)

            precision = overlap / pred_len if pred_len > 0 else 0.0
            recall = overlap / gold_len if gold_len > 0 else 0.0
            f1 = (
                (2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )
            em = 1.0 if (p_start == g_start and p_end == g_end) else 0.0

            em_sum += em
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
        return em_sum, precision_sum, recall_sum, f1_sum

    for epoch in range(int(cfg["train"]["epochs"])):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                start_logits, end_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
                loss = (
                    criterion(start_logits, start_pos) + criterion(end_logits, end_pos)
                ) / 2.0
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg["train"].get("clip_grad", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(cfg["train"]["clip_grad"])
                    )
                optimizer.step()
            total_loss += loss.item()

        # Validation EM
        model.eval()
        with torch.no_grad():
            em_cnt = 0.0
            precision_sum = 0.0
            recall_sum = 0.0
            f1_sum = 0.0
            n = 0
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_pos = batch["start_positions"].to(device)
                end_pos = batch["end_positions"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                    start_logits, end_logits = model(
                        input_ids=input_ids, attention_mask=attention_mask
                    )
                sp = start_logits.argmax(dim=1)
                ep = end_logits.argmax(dim=1)
                batch_em, batch_prec, batch_rec, batch_f1 = _accumulate_span_metrics(
                    sp, ep, start_pos, end_pos
                )
                em_cnt += batch_em
                precision_sum += batch_prec
                recall_sum += batch_rec
                f1_sum += batch_f1
                n += input_ids.size(0)
            em = (em_cnt / n) if n else 0.0
            precision = (precision_sum / n) if n else 0.0
            recall = (recall_sum / n) if n else 0.0
            f1 = (f1_sum / n) if n else 0.0
        # Log per-epoch metric
        try:
            mlflow.log_metric("val_exact_match", float(em), step=epoch)
            mlflow.log_metric("val_precision", float(precision), step=epoch)
            mlflow.log_metric("val_recall", float(recall), step=epoch)
            mlflow.log_metric("val_f1", float(f1), step=epoch)
        except Exception:
            pass

        if em > best_em:
            best_em = em
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    # Save checkpoint (best on val)
    ckpt_path = outdir / "checkpoint.pt"
    torch.save({"state_dict": best_state, "val_exact_match": best_em}, ckpt_path)

    # Evaluate on test with best weights
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        em_cnt = 0.0
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        n = 0
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_pos = batch["start_positions"].to(device)
            end_pos = batch["end_positions"].to(device)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=dtype):
                start_logits, end_logits = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            sp = start_logits.argmax(dim=1)
            ep = end_logits.argmax(dim=1)
            batch_em, batch_prec, batch_rec, batch_f1 = _accumulate_span_metrics(
                sp, ep, start_pos, end_pos
            )
            em_cnt += batch_em
            precision_sum += batch_prec
            recall_sum += batch_rec
            f1_sum += batch_f1
            n += input_ids.size(0)
        test_em = (em_cnt / n) if n else 0.0
        test_precision = (precision_sum / n) if n else 0.0
        test_recall = (recall_sum / n) if n else 0.0
        test_f1 = (f1_sum / n) if n else 0.0

    metrics = {
        "val_exact_match": float(best_em),
        "val_precision": float(best_precision),
        "val_recall": float(best_recall),
        "val_f1": float(best_f1),
        "test_exact_match": float(test_em),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "runtime_s": float(time.time() - start),
        "epochs": int(cfg["train"]["epochs"]),
        "batch_size": int(cfg["train"]["batch_size"]),
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # Log final metrics and artifacts to MLflow
    try:
        mlflow.log_metrics(
            {
                "final_val_exact_match": float(best_em),
                "final_val_precision": float(best_precision),
                "final_val_recall": float(best_recall),
                "final_val_f1": float(best_f1),
                "final_test_exact_match": float(test_em),
                "final_test_precision": float(test_precision),
                "final_test_recall": float(test_recall),
                "final_test_f1": float(test_f1),
                "runtime_s": float(metrics["runtime_s"]),
            }
        )
        mlflow.log_artifact(str(outdir / "config.json"), artifact_path="config")
        mlflow.log_artifact(str(outdir / "metrics.json"), artifact_path="metrics")
        mlflow.log_artifact(str(outdir / "checkpoint.pt"), artifact_path="checkpoints")
    except Exception:
        pass
    finally:
        try:
            mlflow.end_run()
        except Exception:
            pass
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--storage", default="sqlite:///_optuna/dataaug.db")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--disable-augmentation", action="store_true")
    args = ap.parse_args()

    # Connect to SQLite
    storage = args.storage
    if not storage.startswith("sqlite:///"):
        raise SystemExit("Only sqlite storage is supported in this script.")
    db_path = storage.replace("sqlite:///", "")
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # Find study and best trial by objective 0 (desc for MAXIMIZE)
    row = cur.execute(
        "select study_id from studies where study_name=?", (args.study,)
    ).fetchone()
    if not row:
        raise SystemExit(f"Study not found: {args.study}")
    sid = int(row[0])
    dir0 = cur.execute(
        "select direction from study_directions where study_id=? and objective=0",
        (sid,),
    ).fetchone()
    order = "desc" if (dir0 and dir0[0] == "MAXIMIZE") else "asc"
    best = cur.execute(
        f"""
        select t.trial_id, t.number, v.value
        from trials t join trial_values v on v.trial_id=t.trial_id and v.objective=0
        where t.study_id=? and t.state='COMPLETE' and v.value is not null
        order by v.value {order} limit 1
        """,
        (sid,),
    ).fetchone()
    if not best:
        raise SystemExit("No completed trials with objective value found.")

    trial_id, trial_num, best_val = int(best[0]), int(best[1]), float(best[2])
    params = _decode_params(cur, trial_id)
    cfg = _build_cfg_from_params(params)
    if args.disable_augmentation:
        cfg["augmentation"] = {"enabled": False}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"Retraining trial #{trial_num} from study '{args.study}' (val={best_val:.4f})"
    )
    # Configure MLflow for single-train runs
    _setup_mlflow(experiment="evidence-SINGLE")
    metrics = train_and_eval(cfg, outdir)
    print("Saved:")
    print(f"  - {outdir / 'config.json'}")
    print(f"  - {outdir / 'checkpoint.pt'}")
    print(f"  - {outdir / 'metrics.json'}")
    print("Metrics:", json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
