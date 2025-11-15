from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest

from dataaug_multi_both.mlflow_buffer import MlflowBuffer


class DummyFailureMlflow:
    def __getattr__(self, item):  # pragma: no cover - defensive
        raise AttributeError(item)

    @staticmethod
    def log_param(*args, **kwargs):  # pragma: no cover - replaced in test
        raise RuntimeError("forced failure")


def test_mlflow_buffer_records_and_replays(tmp_path, monkeypatch):
    buffer_dir = tmp_path / "buffer"
    buffer = MlflowBuffer(buffer_dir)

    # Force failures by monkeypatching mlflow methods
    monkeypatch.setattr(mlflow, "log_param", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(mlflow, "log_metric", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))
    monkeypatch.setattr(mlflow, "set_tags", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")))

    buffer.log_param("alpha", 1)
    buffer.log_metric("metric", 0.5, step=1)
    buffer.log_tags({"foo": "bar"})

    shards = list(buffer_dir.glob("*.jsonl"))
    assert shards

    # Restore mlflow to capture replayed payloads
    records: list[dict] = []
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: records.append({"type": "param", "key": k, "value": v}))
    monkeypatch.setattr(mlflow, "log_metric", lambda k, v, step=None: records.append({"type": "metric", "key": k, "value": v, "step": step}))
    monkeypatch.setattr(mlflow, "set_tags", lambda tags: records.append({"type": "tags", "tags": tags}))

    buffer.replay()

    assert {r["type"] for r in records} == {"param", "metric", "tags"}
    assert not list(buffer_dir.glob("*.jsonl"))
