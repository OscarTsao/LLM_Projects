from __future__ import annotations

from dataaug_multi_both.hpo import driver


def test_driver_simulation_executes() -> None:
    exit_code = driver.main(
        [
            "--simulate",
            "--trials-a",
            "6",
            "--trials-b",
            "4",
            "--epochs-a",
            "2",
            "--epochs-b",
            "4",
            "--timeout",
            "120",
            "--k-top",
            "2",
        ]
    )
    assert exit_code == 0
