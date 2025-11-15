from __future__ import annotations

from pathlib import Path
import json

from dataaug_multi_both.hpo import retrain


class _StubBuffer:
    def log_metric(self, *args, **kwargs):
        pass

    def log_tags(self, *args, **kwargs):
        pass

    def log_artifact(self, *args, **kwargs):
        pass

    def replay(self):
        pass


def test_retrain_cli(tmp_path, monkeypatch):
    monkeypatch.setattr(retrain, "init_mlflow", lambda *a, **k: _StubBuffer())
    monkeypatch.setattr(retrain, "build_objective", lambda cfg: lambda trial, params, settings: 0.4)
    best_path = tmp_path / "best.json"
    with best_path.open("w", encoding="utf-8") as fh:
        json.dump({"params": {"dummy": 1}}, fh)

    retrain_output = tmp_path / "retrain"
    code = retrain.main(
        [
            "--best-path",
            str(best_path),
        "--seeds",
        "1",
            "--epochs",
            "1",
            "--output-root",
            str(retrain_output),
        "--synthetic-train-size",
        "6",
        "--synthetic-val-size",
        "3",
        "--synthetic-seq-len",
        "32",
        ]
    )
    assert code == 0
    summary = retrain_output / "artifacts/retrain/summary.json"
    assert summary.exists()
