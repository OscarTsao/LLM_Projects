from pathlib import Path

from src.dataaug_multi_both.utils.mlflow_setup import init_mlflow


def test_mlflow_sqlite_tmp(tmp_path):
    uri = init_mlflow(str(tmp_path), "unittest-exp")
    assert uri.startswith("sqlite:///")
    assert (Path(tmp_path) / "mlflow.db").exists()
