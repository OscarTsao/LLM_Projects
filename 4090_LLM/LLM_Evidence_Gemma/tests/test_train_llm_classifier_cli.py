import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest


def test_classifier_cli_cpu_only(tmp_path: Path, monkeypatch):
    data = pd.DataFrame(
        [
            {"text": "happy day sunshine", "label": "pos"},
            {"text": "gloomy night sadness", "label": "neg"},
        ]
    )
    train_csv = tmp_path / "train.csv"
    data.to_csv(train_csv, index=False)

    cmd = [
        "python",
        "src/training/train_llm_classifier.py",
        "--model_name",
        "sshleifer/tiny-gpt2",
        "--mode",
        "causal",
        "--train_file",
        str(train_csv),
        "--validation_file",
        str(train_csv),
        "--text_column",
        "text",
        "--label_column",
        "label",
        "--output_dir",
        str(tmp_path / "outputs"),
        "--num_train_epochs",
        "0.1",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--learning_rate",
        "1e-4",
        "--flash_attn2",
        "false",
        "--cpu_only",
        "--warmup_head_steps",
        "0",
        "--lora_r",
        "4",
        "--pad_to_multiple_of",
        "4",
        "--disable_lora",
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + (":" if env.get("PYTHONPATH") else "") + str(Path.cwd())
    subprocess.run(cmd, check=True, env=env)
    metrics_file = tmp_path / "outputs" / "eval_results.json"
    assert metrics_file.exists()
    metrics = json.loads(metrics_file.read_text())
    assert "eval_macro_f1" in metrics
