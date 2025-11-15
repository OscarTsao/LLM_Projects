from __future__ import annotations

import logging
from typing import List

import pytest

from src.dataaug_multi_both.hpo.trial_executor import TrialExecutor, TrialResult, TrialSpec


class MlflowStub:
    def __init__(self) -> None:
        self.tags: dict[str, str] = {}

    def set_tag(self, key: str, value: str) -> None:
        self.tags[key] = value


@pytest.mark.unit
def test_trial_executor_emits_progress(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    execution_order: List[str] = []
    mlflow_stub = MlflowStub()

    def run_trial(spec: TrialSpec) -> TrialResult:
        trial_id = spec.ensure_id()
        execution_order.append(trial_id)
        return TrialResult(trial_id=trial_id, metric=0.5, status="success", duration_seconds=0.1)

    executor = TrialExecutor(run_trial=run_trial, mlflow_client=mlflow_stub)

    specs = [TrialSpec(config={"lr": 1e-3}), TrialSpec(config={"lr": 5e-4})]
    executor.execute(specs)

    assert len(executor.results) == 2
    assert execution_order == [result.trial_id for result in executor.results]

    # Validate that progress messages were logged for start/end of each trial
    starts = [record for record in caplog.records if "event" in getattr(record, "event", {})]
    assert starts, "Expected progress log records"
    for record in starts:
        payload = record.event
        assert payload["trial_index"] >= 1
        assert payload["trial_total"] == 2

    # MLflow tags should contain progress indicators
    assert mlflow_stub.tags["hpo.trial_total"] == "2"
    assert "hpo.best_metric" in mlflow_stub.tags
