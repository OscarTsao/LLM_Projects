import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: Optional[int] = None) -> int:
    """Set Python, NumPy, and (optionally) Torch seeds.

    Returns the resolved seed for logging.
    """
    if seed is None:
        seed = int(os.environ.get('PY_SEED', '42'))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass
    return seed

