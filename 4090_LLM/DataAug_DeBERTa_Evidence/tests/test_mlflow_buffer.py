from __future__ import annotations

import json
from pathlib import Path

import mlflow

from dataaug_multi_both import mlflow_buffer


def test_mlflow_buffer_stores_and_replays(tmp_path, monkeypatch):
    buffer_dir = tmp_path / "buffer"
    cfg = {"mlflow": {"buffer": {"dir": str(buffer_dir)}}}

    mlflow_buffer._BUFFER = None  # type: ignore[attr-defined]
    mlflow_buffer.configure_buffer(cfg)

    def failing_log_metrics(metrics, step=None):  # pragma: no cover - invoked intentionally
        raise RuntimeError("backend down")

    monkeypatch.setattr(mlflow, "log_metrics", failing_log_metrics)

    mlflow_buffer.log_metric_safe({"accuracy": 0.5}, step=1)
    pending = buffer_dir / "pending.jsonl"
    assert pending.exists()
    events = [json.loads(line) for line in pending.read_text(encoding="utf-8").splitlines() if line]
    assert events and events[0]["type"] == "metrics"

    calls = {"metrics": 0}

    def success_log_metrics(metrics, step=None):
        calls["metrics"] += 1

    monkeypatch.setattr(mlflow, "log_metrics", success_log_metrics)

    mlflow_buffer.on_run_start()
    assert calls["metrics"] == 1
    assert not pending.exists()

    mlflow_buffer._BUFFER = None  # type: ignore[attr-defined]
