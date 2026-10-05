"""In-memory curves and detailed support-selection results."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import FitSettings, validate_covariance, positive_integer
from .penalty import PenaltyConstants
from .selection import BootstrapSelection, PlateauScaleSelection
from .supports.common import validate_support_mask


def _readonly(value, dtype=None):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, eq=False)
class SupportCurve:
    """Raw objective and fitted model at each searched dimension.

    Copies of arrays are read-only. Invalid fits retain Inf objectives and
    NaN parameters. ``omega_star`` is optional provenance, never hidden truth
    used for selection. Diagonals are included in all valid model masks.
    """

    sigma_hat: np.ndarray
    dimensions: np.ndarray
    raw_objectives: np.ndarray
    support_masks: np.ndarray
    support_valid: np.ndarray
    fitted_lambdas: np.ndarray
    fitted_omegas: np.ndarray
    fit_settings: FitSettings
    resolved_omega_ref: float
    allowed_edges: tuple[tuple[int, int], ...]
    support_scope: str = "all"
    nested_supports: bool = True
    num_samples: int | None = None
    omega_star: float | None = None

    def __post_init__(self):
        sigma = validate_covariance(self.sigma_hat)
        n = len(sigma)
        dims = np.asarray(self.dimensions)
        count = len(dims) if dims.ndim == 1 else 0
        if count == 0 or not np.array_equal(dims, np.arange(1, count + 1)):
            raise ValueError("Curve dimensions must be a contiguous prefix starting at 1.")
        allowed = tuple(self.allowed_edges)
        if len(set(allowed)) != len(allowed) or any(len(edge) != 2 or edge[0] == edge[1] or not (0 <= edge[0] < n and 0 <= edge[1] < n) for edge in allowed):
            raise ValueError("allowed_edges must contain unique off-diagonal matrix positions.")
        if count > len(allowed) + 1:
            raise ValueError("Curve dimensions exceed the allowed support space.")
        raw = np.asarray(self.raw_objectives, dtype=float)
        valid = np.asarray(self.support_valid, dtype=bool)
        masks = np.asarray(self.support_masks, dtype=bool)
        lambdas = np.asarray(self.fitted_lambdas, dtype=float)
        omegas = np.asarray(self.fitted_omegas, dtype=float)
        if raw.shape != (count,) or valid.shape != (count,) or omegas.shape != (count,) or masks.shape != (count, n, n) or lambdas.shape != (count, n, n):
            raise ValueError("All curve arrays must match dimensions and covariance size.")
        previous = np.eye(n, dtype=bool)
        failed = False
        for index in range(count):
            if not valid[index]:
                if not np.isposinf(raw[index]):
                    raise ValueError("Invalid fits must have Inf objectives.")
                failed = True
                continue
            if not np.isfinite(raw[index]) or raw[index] < 0 or not np.all(np.isfinite(lambdas[index])) or not np.isfinite(omegas[index]):
                raise ValueError("Valid fits must have finite parameters and nonnegative objectives.")
            mask = validate_support_mask(masks[index], n, int(dims[index]) - 1, allowed)
            if np.any(lambdas[index][~mask] != 0):
                raise ValueError("Fitted Lambda has coefficients outside its selected mask.")
            if self.nested_supports and (failed or np.any(previous & ~mask)):
                raise ValueError("Valid nested supports cannot follow a failed fit or discard edges.")
            previous = mask
        if self.num_samples is not None:
            positive_integer(self.num_samples, "num_samples", minimum=2)
        if not isinstance(self.fit_settings, FitSettings):
            raise TypeError("fit_settings must be a FitSettings instance.")
        for name, value, dtype in (
            ("sigma_hat", sigma, float), ("dimensions", dims, int),
            ("raw_objectives", raw, float), ("support_masks", masks, bool),
            ("support_valid", valid, bool), ("fitted_lambdas", lambdas, float),
            ("fitted_omegas", omegas, float),
        ):
            object.__setattr__(self, name, _readonly(value, dtype))
        object.__setattr__(self, "allowed_edges", allowed)


@dataclass(frozen=True, eq=False)
class SelectionResult:
    """Selected support together with fitting, penalty, and selection evidence."""

    support: np.ndarray
    selected_dimension: int
    selected_edges: tuple[tuple[int, int], ...]
    method: str
    curve: SupportCurve
    num_samples: int
    penalty_constants: PenaltyConstants
    penalty_values: np.ndarray
    lm_values: np.ndarray
    lm_mode: str
    objective_floor: float
    screened_objectives: np.ndarray
    diagnostics: BootstrapSelection | PlateauScaleSelection

    def __post_init__(self):
        for name in ("support", "penalty_values", "lm_values", "screened_objectives"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    @property
    def resolved_omega_ref(self):
        return self.curve.resolved_omega_ref

    @property
    def fitted_lambda(self):
        index = np.flatnonzero(self.curve.dimensions == self.selected_dimension)[0]
        return self.curve.fitted_lambdas[index]

    @property
    def fitted_omega(self):
        index = np.flatnonzero(self.curve.dimensions == self.selected_dimension)[0]
        return float(self.curve.fitted_omegas[index])

    @property
    def bootstrap_comparisons(self):
        return self.diagnostics.comparisons if isinstance(self.diagnostics, BootstrapSelection) else ()
