from __future__ import annotations

from pathlib import Path

import optuna

from dataaug_multi_both.hpo import driver


def test_driver_dry_run() -> None:
    exit_code = driver.main(["--dry-run"])
    assert exit_code == 0
