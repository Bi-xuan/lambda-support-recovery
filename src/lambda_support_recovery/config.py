"""Validated fitting settings and configurable theorem-penalty references."""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .penalty import PenaltyConstants, validate_constants


def positive_integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}.")
    return int(value)


def nonnegative_float(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number.")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number.") from exc
    if not np.isfinite(value) or value < 0 or (positive and value == 0):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be finite and {qualifier}.")
    return value


def validate_covariance(value, *, name="sigma_hat", n=None):
    """Copy a finite, symmetric, positive-definite covariance of size >= 2."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real.")
    try:
        matrix = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric square matrix.") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or len(matrix) < 2:
        raise ValueError(f"{name} must be a square matrix of size at least 2.")
    if n is not None and matrix.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n}).")
    if not np.all(np.isfinite(matrix)) or not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be finite and symmetric.")
    matrix = (matrix + matrix.T) / 2
    if np.linalg.eigvalsh(matrix)[0] <= 0:
        raise ValueError(f"{name} must be positive definite; singular input requires explicit preprocessing.")
    return matrix


@dataclass(frozen=True)
class FitSettings:
    """Settings shared by curve fitting and fixed-support bootstrap refitting."""

    omega_fixed: float = 1.0
    beta: float = 1.0
    max_iter: int = 800
    tol: float = 1e-7
    zero_tol: float = 1e-5
    max_restarts: int = 10
    init_strategy: str = "halton"

    def __post_init__(self):
        for name in ("max_iter", "max_restarts"):
            object.__setattr__(self, name, positive_integer(getattr(self, name), name))
        for name in ("beta", "tol", "zero_tol"):
            object.__setattr__(self, name, nonnegative_float(getattr(self, name), name, positive=name in ("beta", "tol")))
        object.__setattr__(self, "omega_fixed", nonnegative_float(self.omega_fixed, "omega_fixed"))
        if self.init_strategy not in ("halton", "random"):
            raise ValueError("init_strategy must be 'halton' or 'random'.")


@dataclass(frozen=True)
class PenaltyConfig:
    """References for the MS-S theorem penalty, independent of the fitted model.

    ``sigma=None`` uses the observed covariance. A positive scalar means that
    scalar times the identity; a matrix supplies an alternative covariance.
    ``lambda_bound`` is the theorem's L, default 1, and ``noise_bound`` is r.
    The six optional summaries override quantities derived from the reference.
    Despite their names, lambda_* summarize covariance eigenvalues, not entries
    or eigenvalues of the fitted Lambda matrix.
    """

    sigma: float | np.ndarray | None = None
    lambda_bound: float = 1.0
    noise_bound: float = 1.0
    xi: float = 10.0
    lambda_sum: float | None = None
    lambda_inf_norm: float | None = None
    lambda_2_norm: float | None = None
    sigma_fro_norm: float | None = None
    sigma_op_norm: float | None = None
    sigma_trace: float | None = None

    def __post_init__(self):
        nonnegative_float(self.lambda_bound, "lambda_bound", positive=True)
        nonnegative_float(self.noise_bound, "noise_bound", positive=True)
        nonnegative_float(self.xi, "xi", positive=True)
        for name in ("lambda_sum", "lambda_inf_norm", "lambda_2_norm", "sigma_fro_norm", "sigma_op_norm", "sigma_trace"):
            value = getattr(self, name)
            if value is not None:
                nonnegative_float(value, name)


def build_penalty_constants(sigma_hat, num_samples, *, config=None, lm_weight=1.0):
    """Derive usable penalty constants without requiring population truth."""
    observed = validate_covariance(sigma_hat)
    num_samples = positive_integer(num_samples, "num_samples", minimum=2)
    lm_weight = nonnegative_float(lm_weight, "lm_weight", positive=True)
    config = PenaltyConfig() if config is None else config
    if not isinstance(config, PenaltyConfig):
        raise TypeError("penalty_config must be a PenaltyConfig instance.")
    n = len(observed)
    if config.sigma is None:
        reference = observed
    elif np.ndim(config.sigma) == 0:
        reference = np.eye(n) * nonnegative_float(config.sigma, "penalty sigma", positive=True)
    else:
        reference = validate_covariance(config.sigma, name="penalty sigma", n=n)
    eigenvalues = np.linalg.eigvalsh(reference)
    summaries = dict(
        lambda_sum=float(eigenvalues.sum()),
        lambda_inf_norm=float(np.max(np.abs(eigenvalues))),
        lambda_2_norm=float(np.linalg.norm(eigenvalues)),
        sigma_fro_norm=float(np.linalg.norm(reference, "fro")),
        sigma_op_norm=float(np.linalg.norm(reference, 2)),
        sigma_trace=float(np.trace(reference)),
    )
    for name in summaries:
        override = getattr(config, name)
        if override is not None:
            summaries[name] = nonnegative_float(override, name)
    constants = PenaltyConstants(
        num_samples=num_samples, n=n, Lm=lm_weight,
        L=float(config.lambda_bound), r=float(config.noise_bound), xi=float(config.xi),
        **summaries,
    )
    validate_constants(constants)
    if constants.lambda_sum == 0 and constants.lambda_inf_norm == 0 and constants.lambda_2_norm == 0 and constants.sigma_op_norm == 0:
        raise ValueError("Penalty references must produce a positive theorem scale.")
    return constants
