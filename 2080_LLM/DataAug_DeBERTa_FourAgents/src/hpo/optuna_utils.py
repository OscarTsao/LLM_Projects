from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class BestState:
    best_value: Optional[float] = None
    best_trial_id: Optional[int] = None
    last_improved_idx: int = -1


def update_best_state(state: BestState, value: float, trial_number: int, direction: str = "maximize") -> bool:
    improved = False
    if state.best_value is None:
        improved = True
    else:
        if direction == "maximize":
            improved = value > state.best_value + 1e-12
        else:
            improved = value < state.best_value - 1e-12
    if improved:
        state.best_value = value
        state.best_trial_id = trial_number
        state.last_improved_idx = trial_number
    return improved


def estimate_eta(start_time: float, completed: int, total: int) -> float:
    if completed <= 0:
        return float("inf")
    elapsed = time.time() - start_time
    avg = elapsed / completed
    remaining = max(total - completed, 0)
    return remaining * avg


def format_progress(
    trial_idx: int,
    trial_total: int,
    completed_or_pruned: int,
    started: int,
    study_elapsed: float,
    eta_seconds: float,
    best_value: Optional[float],
    best_trial_id: Optional[int],
) -> str:
    cr = 0.0 if started == 0 else completed_or_pruned / float(started)
    best_str = "none" if best_value is None else f"{best_value:.6f} (trial {best_trial_id})"
    return (
        f"trial {trial_idx}/{trial_total} | completion_rate={cr:.2%} | elapsed={study_elapsed:.1f}s "
        f"| eta={eta_seconds:.1f}s | best={best_str}"
    )

