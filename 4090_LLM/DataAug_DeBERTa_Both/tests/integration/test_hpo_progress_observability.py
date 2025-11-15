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


@pytest.mark.integration
def test_hpo_progress_observability(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    execution_order: List[str] = []
    mlflow_stub = MlflowStub()

    def run_trial(spec: TrialSpec) -> TrialResult:
        trial_id = spec.ensure_id()
        execution_order.append(trial_id)
        metric = spec.config.get("metric", 0.0)
        return TrialResult(trial_id=trial_id, metric=metric, status="success", duration_seconds=0.05)

    executor = TrialExecutor(run_trial=run_trial, mlflow_client=mlflow_stub)

    specs = [
        TrialSpec(config={"metric": 0.72}),
        TrialSpec(config={"metric": 0.81}),
        TrialSpec(config={"metric": 0.78}),
    ]

    results = executor.execute(specs)

    assert len(results) == 3
    assert execution_order == [result.trial_id for result in results]
    assert max(result.metric for result in results if result.metric is not None) == pytest.approx(0.81)

    progress_records = [record for record in caplog.records if getattr(record, "component", "") == "hpo"]
    assert progress_records, "Expected hpo progress logs"
    assert any("completion_rate" in record.event for record in progress_records)

    # Final MLflow tags should reflect best metric and completion state
    assert mlflow_stub.tags["hpo.best_metric"] == "0.81"
    assert mlflow_stub.tags["hpo.trial_index"] == "3"
    assert mlflow_stub.tags["hpo.trial_total"] == "3"
