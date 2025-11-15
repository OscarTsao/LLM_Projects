import json
import shutil
import sys
from pathlib import Path

import pytest

from src.pipeline import run_pipeline

DATA_PATH = Path("data/redsm5_sample.jsonl")


def run_pipeline_with_tmp(tmp_path: Path):
    pipeline_cfg = tmp_path / "pipeline.yaml"
    pipeline_cfg.write_text(
        "outputs:\n  root: {}\n".format((tmp_path / "outputs").as_posix())
    )

    data_cfg = tmp_path / "data.yaml"
    data_cfg.write_text(
        "dataset:\n  path: {}\n".format(DATA_PATH.as_posix())
    )

    args = [
        "run_pipeline",
        "--data-config",
        str(data_cfg),
        "--pipeline-config",
        str(pipeline_cfg),
        "--calibration-path",
        str(tmp_path / "artifacts" / "calibration.json"),
        "--seed",
        "123",
    ]
    old_argv = sys.argv
    sys.argv = args
    try:
        run_pipeline.main()
    finally:
        sys.argv = old_argv
    return tmp_path / "outputs"


def test_pipeline_smoke(tmp_path):
    outputs_root = run_pipeline_with_tmp(tmp_path)
    runs = sorted((outputs_root / "evaluation").glob("run_*"))
    assert runs, "Expected evaluation run directory"
    latest = runs[-1]
    metrics_path = latest / "test_metrics.json"
    assert metrics_path.is_file()
    data = json.loads(metrics_path.read_text())
    assert data["criteria_auroc"] >= 0.8
    assert data["negation_precision"] >= 0.9
    # Clean generated mlruns to keep repo tidy during tests
    shutil.rmtree("mlruns", ignore_errors=True)
