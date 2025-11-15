from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from psya_agent.data import build_examples, load_annotations, load_posts, split_examples
from psya_agent.features import EvalQADataset, eval_collate_fn, prepare_eval_features
from psya_agent.metrics import save_metrics
from psya_agent.modeling import SpanBertForQuestionAnswering
from psya_agent.train_utils import evaluate, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved SpanBERT QA checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the YAML config used for training.")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the model checkpoint (.pt) produced during training.",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="test",
        help="Dataset split to evaluate against.",
    )
    parser.add_argument(
        "--metrics_output",
        default=None,
        help="Optional path to save evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--predictions_output",
        default=None,
        help="Optional path to save span predictions as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")

    cfg = OmegaConf.load(args.config)
    device = select_device(cfg.training.device)
    local_files_only = bool(cfg.model.local_files_only) if "local_files_only" in cfg.model else False
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    posts = load_posts(Path(cfg.data.groundtruth_path))
    annotations = load_annotations(Path(cfg.data.annotations_path), positive_only=cfg.data.positive_only)
    examples = build_examples(posts, annotations)
    train_examples, val_examples, test_examples = split_examples(
        examples, cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.seed
    )

    target_examples = val_examples if args.split == "val" else test_examples
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        use_fast=True,
        local_files_only=local_files_only,
    )
    features = prepare_eval_features(target_examples, tokenizer, cfg.features.max_length, cfg.features.doc_stride)
    dataset = EvalQADataset(features)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=eval_collate_fn,
    )

    model = SpanBertForQuestionAnswering(
        cfg.model.pretrained_model_name_or_path,
        dropout=cfg.model.dropout,
        local_files_only=local_files_only,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint)
    model.to(device)

    metrics, predictions = evaluate(
        model,
        dataloader,
        target_examples,
        features,
        cfg.features.n_best_size,
        cfg.features.max_answer_length,
        device,
        return_predictions=True,
    )

    logging.info("Split %s metrics: %s", args.split, metrics)
    if args.metrics_output:
        save_metrics(metrics, Path(args.metrics_output))
    if args.predictions_output:
        Path(args.predictions_output).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.predictions_output).open("w", encoding="utf-8") as fh:
            json.dump(predictions, fh, indent=2)


if __name__ == "__main__":
    main()
