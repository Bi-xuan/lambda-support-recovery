"""Lambda support recovery from an empirical covariance matrix."""

from .api import select_support, select_from_curve
from .config import FitSettings, PenaltyConfig, build_penalty_constants
from .curve import compute_support_curve
from .penalty import PenaltyConstants
from .results import SelectionResult, SupportCurve

__version__ = "0.1.0"

__all__ = [
    "select_support", "compute_support_curve", "select_from_curve",
    "PenaltyConfig", "PenaltyConstants", "build_penalty_constants",
    "FitSettings", "SupportCurve", "SelectionResult",
]
