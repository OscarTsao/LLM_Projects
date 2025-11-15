#!/usr/bin/env python3
"""
Generate test-set predictions for the best Evidence model (no augmentation).

This script:
- Loads the retrained best no-aug Evidence checkpoint from outputs/retrain
- Reproduces the exact deterministic 80/10/10 split used during retrain
- Runs batched inference to extract predicted spans
- Writes a CSV with: post_id, post, evidence_answer, predicted_span

Default checkpoint directory: outputs/retrain/evidence_noaug_best_e100
Fallback checkpoint directory: outputs/retrain/evidence_noaug_best

Usage:
  python scripts/predict_evidence_test.py \
      --ckpt-dir outputs/retrain/evidence_noaug_best_e100 \
      --out-csv outputs/predictions/evidence_noaug_best_e100_test_predictions.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoTokenizer


def load_cfg_and_ckpt_dir(preferred: Path, fallback: Path) -> tuple[dict[str, Any], Path]:
    if (preferred / "config.json").is_file() and (preferred / "checkpoint.pt").is_file():
        cfg = json.loads((preferred / "config.json").read_text())
        return cfg, preferred
    if (fallback / "config.json").is_file() and (fallback / "checkpoint.pt").is_file():
        cfg = json.loads((fallback / "config.json").read_text())
        return cfg, fallback
    raise FileNotFoundError(
        "Could not find config.json and checkpoint.pt in either "
        f"'{preferred}' or '{fallback}'."
    )


def get_test_indices(n: int, seed: int) -> list[int]:
    # Deterministic 80/10/10 split, matching scripts/retrain_best_evidence.py
    tr = int(0.8 * n)
    va = int(0.1 * n)
    te = n - tr - va
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=gen).tolist()
    return idx[tr + va : tr + va + te]


def build_model(model_name: str, head_cfg: dict[str, Any] | None):
    # Import Model lazily to avoid circular imports
    # Ensure 'src' is on sys.path for module resolution
    project_root = Path(__file__).resolve().parent.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from Project.Evidence.models.model import Model as EvidenceModel

    return EvidenceModel(model_name=model_name, head_cfg=head_cfg)


def run_predictions(
    ckpt_dir: Path, cfg: dict[str, Any], out_csv: Path, batch_size: int = 16
) -> None:
    # Paths and config
    data_csv = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "processed"
        / "redsm5_matched_evidence.csv"
    )
    if not data_csv.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_csv}")

    model_name = cfg["model"]["name"]
    tok_max_len = int(cfg["tok"]["max_length"]) if cfg.get("tok") else 512
    seed = int(cfg.get("meta", {}).get("seed", 42))

    # Load tokenizer and raw CSV rows for metadata
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    with data_csv.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        rows = list(reader)
    if not rows:
        raise ValueError("No rows found in dataset CSV.")

    # Compute test indices deterministically
    test_indices = get_test_indices(len(rows), seed)

    # Gather contexts and gold metas in the same order
    contexts: list[str] = []
    gold_info: list[dict[str, str]] = []
    for i in test_indices:
        r = rows[i]
        contexts.append(r["post_text"])  # context/post
        gold_info.append(
            {
                "post_id": r.get("post_id", ""),
                "post": r.get("post_text", ""),
                "evidence_answer": r.get("sentence_text", ""),
            }
        )

    # Build model and load checkpoint
    model = build_model(model_name=model_name, head_cfg=cfg.get("head", {}))
    ckpt = torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    state_dict = ckpt.get("state_dict")
    if not isinstance(state_dict, dict):
        raise RuntimeError("Invalid checkpoint: missing 'state_dict'")
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # Batched inference with offsets for span text decoding
    preds: list[str] = []
    for s in range(0, len(contexts), batch_size):
        batch_ctx = contexts[s : s + batch_size]
        enc = tokenizer(
            batch_ctx,
            padding="max_length",
            truncation=True,
            max_length=tok_max_len,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            start_logits, end_logits = model(**enc)
            start_pred = start_logits.argmax(dim=-1).cpu()
            end_pred = end_logits.argmax(dim=-1).cpu()

        for ctx, s_idx, e_idx, off in zip(batch_ctx, start_pred, end_pred, offsets, strict=False):
            st = int(s_idx.item())
            en = int(e_idx.item())
            off_pairs = off.tolist()
            if st >= len(off_pairs):
                st = len(off_pairs) - 1
            if en >= len(off_pairs):
                en = len(off_pairs) - 1
            en = max(en, st)
            ch_s = off_pairs[st][0]
            ch_e = off_pairs[en][1]
            span = ctx[ch_s:ch_e] if ch_e > ch_s else ""
            preds.append(span)

    # Write CSV with required columns
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=["post_id", "post", "evidence_answer", "predicted_span"]
        )
        writer.writeheader()
        for meta, pred in zip(gold_info, preds, strict=False):
            writer.writerow({**meta, "predicted_span": pred})

    print(f"Wrote predictions: {out_csv} ({len(preds)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--ckpt-dir",
        type=Path,
        default=Path("outputs/retrain/evidence_noaug_best_e100"),
        help="Directory containing config.json and checkpoint.pt",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path(
            "outputs/predictions/evidence_noaug_best_e100_test_predictions.csv"
        ),
        help="Path to write predictions CSV",
    )
    args = ap.parse_args()

    preferred = args.ckpt_dir
    fallback = Path("outputs/retrain/evidence_noaug_best")
    cfg, use_dir = load_cfg_and_ckpt_dir(preferred, fallback)
    run_predictions(use_dir, cfg, args.out_csv)


if __name__ == "__main__":
    main()
