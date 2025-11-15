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
        print("[evaluate_study] jsonschema not installed; skipping schema validation.")
    except Exception as e:  # pragma: no cover
        print(f"[evaluate_study] Schema validation warning: {e}")


def build_summary(
    study_id: str,
    best_trial_id: str,
    optimization_metric_name: str | None = None,
    best_validation_score: float | None = None,
    base_dir: str = "experiments",
) -> dict:
    report_id = str(uuid.uuid4())
    study_dir = os.path.join(base_dir, f"study_{study_id}")
    trial_report_path = os.path.join(base_dir, f"trial_{best_trial_id}", "evaluation_report.json")

    # Try to fill missing fields from the per-trial report if available
    if os.path.exists(trial_report_path):
        try:
            with open(trial_report_path, encoding="utf-8") as f:
                trial_report = json.load(f)
            if optimization_metric_name is None:
                optimization_metric_name = trial_report.get("optimization_metric_name")
            if best_validation_score is None:
                best_validation_score = trial_report.get("best_validation_score")
        except Exception as e:  # pragma: no cover
            print(f"[evaluate_study] Warning: could not read trial report: {e}")

    payload = {
        "report_id": report_id,
        "study_id": study_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "best_trial_id": best_trial_id,
        "best_validation_score": best_validation_score
        if best_validation_score is not None
        else 0.0,
        "best_trial_report_path": trial_report_path,
        "optimization_metric_name": optimization_metric_name or "val_f1_macro",
        # Optional fields left for future enrichment:
        # "trials_count": 0,
        # "top_trials": [],
    }

    os.makedirs(study_dir, exist_ok=True)
    payload["report_file_path"] = os.path.join(study_dir, "summary_report.json")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate optional study-level summary report.")
    parser.add_argument("--study-id", required=True, help="Study UUID")
    parser.add_argument(
        "--best-trial-id",
        required=True,
        help="Best trial UUID (must correspond to a per-trial report)",
    )
    parser.add_argument(
        "--optimization-metric-name",
        default=None,
        help="Optimization metric name (fallback if not found in per-trial report)",
    )
    parser.add_argument(
        "--best-validation-score",
        type=float,
        default=None,
        help="Best validation score (fallback if not found in per-trial report)",
    )
    parser.add_argument(
        "--experiments-dir",
        default="experiments",
        help="Base experiments directory",
    )
    parser.add_argument(
        "--schema",
        default="specs/002-storage-optimized-training/contracts/study_output_schema.json",
        help="Path to study summary JSON schema",
    )

    args = parser.parse_args(argv)

    payload = build_summary(
        study_id=args.study_id,
        best_trial_id=args.best_trial_id,
        optimization_metric_name=args.optimization_metric_name,
        best_validation_score=args.best_validation_score,
        base_dir=args.experiments_dir,
    )

    # Optional schema validation
    _maybe_validate(args.schema, payload)

    # Write file
    out_path = payload["report_file_path"]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    evaluation_dir = Path("outputs") / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    mirror_path = evaluation_dir / f"summary_{payload['study_id']}.json"
    with mirror_path.open("w", encoding="utf-8") as f:
        json.dump(payload | {"report_file_path": str(mirror_path)}, f, indent=2)
    print(out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
