"""AI/ML Experiment Template - Source Package."""

__version__ = "0.1.0"

# Keep surface minimal in template state to avoid import errors.
# Expose existing subpackages explicitly as they are added.
from . import Criteria  # noqa: F401
from . import Evidence  # noqa: F401
from . import Joint  # noqa: F401
from . import Share  # noqa: F401
from . import utils  # noqa: F401

__all__ = [
    "Criteria",
    "Evidence",
    "Joint",
    "Share",
    "utils",
    "__version__",
]
