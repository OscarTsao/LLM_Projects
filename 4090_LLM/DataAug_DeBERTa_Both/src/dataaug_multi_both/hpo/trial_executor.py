from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrialSpec:
    config: dict
    trial_id: str | None = None

    def ensure_id(self) -> str:
        if self.trial_id is None:
            self.trial_id = str(uuid.uuid4())
        return self.trial_id


@dataclass(slots=True)
class TrialResult:
    trial_id: str
    metric: float | None
    status: str
    duration_seconds: float


class TrialExecutor:
    """Execute Optuna-style trials sequentially and emit progress observability signals."""

    def __init__(
        self,
        run_trial: Callable[[TrialSpec], TrialResult],
        mlflow_client: Optional[object] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.run_trial = run_trial
        self.logger = logger or LOGGER
        self.mlflow_client = mlflow_client
        self._results: List[TrialResult] = []

    @property
    def results(self) -> Sequence[TrialResult]:
        return tuple(self._results)

    def execute(self, trials: Iterable[TrialSpec]) -> Sequence[TrialResult]:
        materialized = list(trials)
        total = len(materialized)
        if total == 0:
            self.logger.warning("No trials provided to TrialExecutor; nothing to execute.")
            return ()

        self._results.clear()
        best_metric: float | None = None
        best_trial_id: str | None = None
        start_wall = time.monotonic()

        for index, spec in enumerate(materialized, start=1):
            trial_id = spec.ensure_id()
            self._log_progress(
                stage="start",
                index=index,
                total=total,
                best_metric=best_metric,
                best_trial_id=best_trial_id,
                start_wall=start_wall,
            )

            trial_start = time.monotonic()
            try:
                result = self.run_trial(spec)
            except Exception as exc:  # pragma: no cover - defensive
                duration = time.monotonic() - trial_start
                result = TrialResult(
                    trial_id=trial_id,
                    metric=None,
                    status=f"failed:{exc.__class__.__name__}",
                    duration_seconds=duration,
                )
                self.logger.exception("Trial %s failed: %s", trial_id, exc)

            result.duration_seconds = getattr(result, "duration_seconds", time.monotonic() - trial_start)
            self._results.append(result)

            if result.metric is not None:
                if best_metric is None or result.metric > best_metric + 1e-12:
                    best_metric = result.metric
                    best_trial_id = result.trial_id

            self._log_progress(
                stage="end",
                index=index,
                total=total,
                best_metric=best_metric,
                best_trial_id=best_trial_id,
                start_wall=start_wall,
            )

        return self.results

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------
    def _log_progress(
        self,
        stage: str,
        index: int,
        total: int,
        best_metric: Optional[float],
        best_trial_id: Optional[str],
        start_wall: float,
    ) -> None:
        completion_rate = index / total
        elapsed = time.monotonic() - start_wall
        eta_seconds = self._estimate_eta(elapsed, index, total)
        payload = {
            "stage": stage,
            "event": stage,
            "trial_index": index,
            "trial_total": total,
            "completion_rate": round(completion_rate, 3),
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": None if eta_seconds is None else round(eta_seconds, 3),
            "best_metric": None if best_metric is None else round(best_metric, 6),
            "best_trial_id": best_trial_id,
        }
        self.logger.info("HPO progress", extra={"component": "hpo", "event": payload})
        self._emit_mlflow_tags(payload)

    def _emit_mlflow_tags(self, payload: dict) -> None:
        if self.mlflow_client is not None and hasattr(self.mlflow_client, "set_tag"):
            for key, value in payload.items():
                tag_key = f"hpo.{key}"
                self.mlflow_client.set_tag(tag_key, "" if value is None else str(value))

    def _estimate_eta(self, elapsed: float, completed: int, total: int) -> Optional[float]:
        if completed == 0:
            return None
        avg_duration = elapsed / completed
        remaining = total - completed
        if remaining <= 0:
            return 0.0
        return avg_duration * remaining


__all__ = ["TrialExecutor", "TrialSpec", "TrialResult"]
