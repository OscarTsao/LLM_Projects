from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_hpo_stage_smoke(tmp_path: Path) -> None:
    env = os.environ.copy()
    env.update(
        {
            "HPO_SMOKE_MODE": "1",
            "HPO_OUTDIR": str(tmp_path),
            "HPO_TRIALS_S0": "2",
            "HPO_TRIALS_S1": "2",
            "HPO_TRIALS_S2": "2",
            "HPO_EPOCHS_S0": "1",
            "HPO_EPOCHS_S1": "1",
            "HPO_EPOCHS_S2": "1",
            "HPO_REFIT_EPOCHS": "1",
            "HPO_SEEDS": "1",
        }
    )
    subprocess.run(
        [
            "python",
            "scripts/run_hpo_stage.py",
            "--agent",
            "criteria",
            "--stage0-trials",
            "2",
            "--stage1-trials",
            "2",
            "--stage2-trials",
            "2",
            "--stage0-epochs",
            "1",
            "--stage1-epochs",
            "1",
            "--stage2-epochs",
            "1",
            "--refit-epochs",
            "1",
            "--seeds",
            "1",
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
    )

    stage0_path = tmp_path / "topk" / "criteria_noaug-criteria-stage0_topk.json"
    stage2_path = tmp_path / "topk" / "criteria_noaug-criteria-stage2_topk.json"
    assert stage0_path.is_file()
    assert stage2_path.is_file()

    data = json.loads(stage2_path.read_text())
    assert data, "Stage 2 should record at least one trial"
