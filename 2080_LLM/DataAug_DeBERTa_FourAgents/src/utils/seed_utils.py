from __future__ import annotations

import hashlib
import os
import random
from typing import Optional


def choose_trial_seed(study_name: str, trial_number: int, base_seed: int = 42) -> int:
    key = f"{study_name}:{trial_number}:{base_seed}".encode("utf-8")
    h = hashlib.sha256(key).hexdigest()
    # 32-bit space for frameworks that expect uint32
    return int(h[:8], 16)


def set_all_seeds(seed: Optional[int]) -> int:
    if seed is None:
        seed = 42
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed

