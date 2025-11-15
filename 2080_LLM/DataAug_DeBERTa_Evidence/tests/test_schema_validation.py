from __future__ import annotations

import json
from pathlib import Path

import optuna
from jsonschema import validate

from dataaug_multi_both.evaluation.reporting import summarize_study_results


def test_study_summary_matches_schema(tmp_path):
    storage_path = tmp_path / "optuna.db"
    storage_uri = f"sqlite:///{storage_path}"
    study = optuna.create_study(storage=storage_uri, study_name="schema-test", direction="maximize")

    def objective(trial: optuna.Trial) -> float:
        trial.set_user_attr("evaluation_report_path", "experiments/trial_dummy/evaluation_report.json")
        return trial.suggest_float("x", 0.0, 1.0)

    study.optimize(objective, n_trials=3)

    schema_path = Path("specs/002-storage-optimized-training/contracts/study_output_schema.json")
    output_path = tmp_path / "summary.json"
    summarize_study_results("schema-test", storage_uri, schema_path, output_path, top_k=2)

    summary = json.loads(output_path.read_text())
    schema = json.loads(schema_path.read_text())
    validate(instance=summary, schema=schema)
