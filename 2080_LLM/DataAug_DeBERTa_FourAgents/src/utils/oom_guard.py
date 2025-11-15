from __future__ import annotations

import contextlib
from typing import Callable


class OOMDuringTraining(RuntimeError):
    pass


def maybe_prune_for_oom(batch_size: int, max_length: int, safety_limit: int = 512 * 64) -> None:
    """Heuristic guard to pre-empt OOM for too-large batch x length.

    If batch_size * max_length exceeds safety_limit, raise an OOM-like error.
    This is a conservative check for single-GPU scenarios.
    """
    try:
        total = int(batch_size) * int(max_length)
        if total > safety_limit:
            raise OOMDuringTraining(
                f"Heuristic OOM: batch_size({batch_size}) x max_length({max_length}) > {safety_limit}"
            )
    except Exception:
        # If inputs invalid, let training validate separately
        pass


@contextlib.contextmanager
def catch_oom() -> Callable[[], None]:
    """Context manager that converts CUDA OOM exceptions to OOMDuringTraining.

    Use around training loops to normalize OOM handling for Optuna pruning.
    """
    try:
        yield
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg and "memory" in msg:
            raise OOMDuringTraining(msg)
        raise

