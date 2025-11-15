import contextlib
import os
import random


def set_seed(
    seed: int = 42, deterministic: bool = True, env_var: str | None = None
) -> int:
    """Set RNG seeds for Python, NumPy, and PyTorch (if available).

    - If `env_var` is provided and set, it overrides the `seed`.
    - When `deterministic` is True, enables PyTorch deterministic algorithms when possible.
    Returns the final seed used.
    """
    if env_var and (v := os.getenv(env_var)):
        with contextlib.suppress(ValueError):
            seed = int(v)

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        np = None  # type: ignore

    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        torch = None  # type: ignore

    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            with contextlib.suppress(Exception):
                torch.use_deterministic_algorithms(True)
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except Exception:
                pass

    return seed
