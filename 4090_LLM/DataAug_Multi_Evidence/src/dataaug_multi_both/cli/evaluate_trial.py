#!/usr/bin/env python
"""
Lightweight CLI to generate a per-trial evaluation report skeleton.

This is intended to make `make evaluate-trial TRIAL_ID=<uuid>` usable in
restricted or offline environments, by producing a schema-conformant JSON
report with placeholder metrics and paths. Integrations that consume this
report can validate file plumbing without requiring a full evaluation run.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path


def _maybe_validate(schema_path: str, payload: dict) -> None:
    try:
        import jsonschema  # type: ignore

        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        jsonschema.validate(instance=payload, schema=schema)
    except ModuleNotFoundError:
        print("[evaluate_trial] jsonschema not installed; skipping schema validation.")
    except Exception as e:  # pragma: no cover
        print(f"[evaluate_trial] Schema validation warning: {e}")


def build_report(trial_id: str, base_dir: str = "outputs/evaluation") -> dict:
    report_id = str(uuid.uuid4())
    trial_dir = os.path.join(base_dir, f"trial_{trial_id}")
    report_path = os.path.join(trial_dir, "evaluation_report.json")

    # Minimal, schema-conformant payload with placeholder values
    payload: dict = {
        "report_id": report_id,
        "trial_id": trial_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {"placeholder": True},
        "optimization_metric_name": "val_f1_macro",
        "best_validation_score": 0.0,
        "evaluated_checkpoints": [
            {
                "checkpoint_id": str(uuid.uuid4()),
                "path": f"experiments/trial_{trial_id}/checkpoints/best.pt",
                "epoch": 0,
                "step": 0,
                "validation_metric": 0.0,
                "co_best": False,
            }
        ],
        "test_metrics": {
            "evidence_binding": {
                "span_f1": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "exact_match": 0.0,
                "char_f1": 0.0,
                "null_span_accuracy": 0.0,
            }
        },
    }

    os.makedirs(trial_dir, exist_ok=True)
    payload["report_file_path"] = report_path
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate per-trial evaluation report.")
    parser.add_argument("--trial-id", required=True, help="Trial UUID")
    parser.add_argument(
        "--experiments-dir",
        default="outputs/evaluation",
        help="Base directory for evaluation artifacts",
    )
    parser.add_argument(
        "--schema",
        default="specs/002-storage-optimized-training/contracts/trial_output_schema.json",
        help="Path to trial report JSON schema",
    )

    args = parser.parse_args(argv)

    payload = build_report(trial_id=args.trial_id, base_dir=args.experiments_dir)

    # Optional schema validation
    _maybe_validate(args.schema, payload)

    # Write file
    out_path = payload["report_file_path"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
