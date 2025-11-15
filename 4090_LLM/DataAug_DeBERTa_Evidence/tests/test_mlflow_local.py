from __future__ import annotations

from pathlib import Path

from dataaug_multi_both.training.train_loop import run_training_job


def test_mlflow_run_creates_artifacts(test_config):
    result = run_training_job(test_config)

    tracking_uri = test_config["mlflow"]["tracking_uri"]
    assert tracking_uri.startswith("sqlite:///"), tracking_uri
    db_path = Path(tracking_uri.replace("sqlite:///", ""))
    assert db_path.exists()

    report_path = Path(result["evaluation_report_path"])
    assert report_path.exists()

    mlruns_dir = Path(test_config["mlflow"]["artifact_location"])
    buffer_dir = Path(test_config["mlflow"]["buffer"]["dir"])
    assert buffer_dir.exists()
