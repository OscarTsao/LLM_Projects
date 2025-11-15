"""
Quick smoke test to overfit a tiny dataset using the LLM classifier CLI.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pandas as pd
import torch

DATA_DIR = Path("data")
SMOKE_CSV = DATA_DIR / "smoke_classifier.csv"

SAMPLES = [
    {"text": "I feel hopeful and energetic today", "label": "positive"},
    {"text": "The weather is gloomy and I feel sad", "label": "negative"},
    {"text": "Working hard makes me proud", "label": "positive"},
    {"text": "Nothing seems to go my way lately", "label": "negative"},
]


def prepare_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(SAMPLES)
    df.to_csv(SMOKE_CSV, index=False)
    return SMOKE_CSV


def run_smoke():
    csv_path = prepare_dataset()
    cmd = [
        "python",
        "src/training/train_llm_classifier.py",
        "--model_name",
        "sshleifer/tiny-gpt2",
        "--mode",
        "causal",
        "--train_file",
        str(csv_path),
        "--validation_file",
        str(csv_path),
        "--text_column",
        "text",
        "--label_column",
        "label",
        "--output_dir",
        "outputs/smoke_classifier",
        "--num_train_epochs",
        "1",
        "--per_device_train_batch_size",
        "1",
        "--per_device_eval_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--lora_r",
        "4",
        "--max_length",
        "128",
        "--warmup_head_steps",
        "0",
        "--flash_attn2",
        "false",
    ]
    cpu_only_env = os.getenv("CPU_ONLY")
    force_cpu = cpu_only_env and cpu_only_env.lower() not in {"0", "false", "no"}
    if force_cpu or not torch.cuda.is_available():
        cmd.append("--cpu_only")
        cmd.append("--disable_lora")
    else:
        cmd.append("--bf16")
    env = os.environ.copy()
    env["PYTHONPATH"] = env.get("PYTHONPATH", "") + (":" if env.get("PYTHONPATH") else "") + str(Path.cwd())
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    run_smoke()
