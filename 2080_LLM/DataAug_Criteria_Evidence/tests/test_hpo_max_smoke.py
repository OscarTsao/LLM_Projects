from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hpo_max_smoke(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HPO_SMOKE_MODE": "1",
            "HPO_OUTDIR": str(tmp_path),
            "HPO_TRIALS": "2",
            "HPO_EPOCHS": "1",
            "HPO_SEEDS": "1",
        }
    )
    subprocess.run(
        [
            "python",
            "scripts/tune_max.py",
            "--agent",
            "criteria",
            "--trials",
            "2",
            "--epochs",
            "1",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    topk_json = tmp_path / "topk" / "criteria_noaug-criteria-max_topk.json"
    assert topk_json.is_file()
    payload = json.loads(topk_json.read_text())
    assert payload, "Expected at least one trial in Top-K output"
