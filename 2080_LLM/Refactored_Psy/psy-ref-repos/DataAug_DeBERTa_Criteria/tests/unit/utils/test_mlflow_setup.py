from __future__ import annotations

from pathlib import Path

import mlflow

from src.dataaug_multi_both.utils.mlflow_setup import BufferedMLflowLogger, init_mlflow


def test_init_mlflow_uses_sqlite_backend(tmp_path: Path):
    tracking_uri = init_mlflow(str(tmp_path), "test_experiment")
    assert tracking_uri.startswith("sqlite:///")
    assert (tmp_path / "mlflow.db").exists()
    assert (tmp_path / "artifacts").is_dir()


def test_buffered_logger_creates_buffer_on_failure(tmp_path: Path, monkeypatch):
    run_dir = tmp_path / "run"
    logger = BufferedMLflowLogger(run_dir)
    monkeypatch.setattr(mlflow, "log_params", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    logger.log_params({"foo": "bar"})
    buffer_file = run_dir / "mlflow_buffer.jsonl"
    assert buffer_file.exists()
    assert buffer_file.read_text(encoding="utf-8").strip() != ""
