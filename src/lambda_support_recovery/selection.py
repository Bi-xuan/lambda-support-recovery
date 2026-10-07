"""Plain Plateau and Plateau_Bootstrap dimension selection.

Both procedures use the exact lower envelope of penalized objectives and rank
bounded plateaus by absolute log-width. Plain Plateau selects at the widest
plateau's geometric center. Plateau_Bootstrap compares
the top plateau models using sequential fixed-support bootstrap tests.
"""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from math import log, sqrt

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import wishart

from .config import FitSettings
from .optimizers.support_search import solve_support_with_restarts
from .supports.common import off_diagonal_edges, validate_support_mask


@dataclass(frozen=True)
class DimensionPath:
    """Exact selected-dimension path over nonnegative penalty scales.

    ``dimensions[i]`` is selected on the interval beginning at
    ``breakpoints[i]``.  The first breakpoint is always zero, and the final
    interval extends to infinity.
    """

    breakpoints: tuple[float, ...]
    dimensions: tuple[int | float, ...]

    def __post_init__(self) -> None:
        if not self.breakpoints or self.breakpoints[0] != 0.0:
            raise ValueError("The first dimension-path breakpoint must be zero.")
        if len(self.breakpoints) != len(self.dimensions):
            raise ValueError("breakpoints and dimensions must have the same length.")
        if any(not np.isfinite(value) for value in self.breakpoints):
            raise ValueError("Dimension-path breakpoints must be finite.")
        if any(value < 0.0 for value in self.breakpoints):
            raise ValueError("Dimension-path breakpoints must be nonnegative.")
        if any(
            right <= left
            for left, right in zip(self.breakpoints, self.breakpoints[1:])
        ):
            raise ValueError("Dimension-path breakpoints must be strictly increasing.")

    @property
    def transition_scales(self) -> tuple[float, ...]:
        """Return the positive scales at which the selected dimension changes."""

        return self.breakpoints[1:]

    def dimension_at(self, scale: float) -> int | float:
        """Return the dimension selected at a finite, nonnegative scale."""

        scale = _finite_nonnegative_float(scale, "scale")
        index = bisect_right(self.breakpoints, scale) - 1
        return self.dimensions[index]


@dataclass(frozen=True)
class PlateauScaleSelection:
    """Penalty scale and dimension selected by the plain Plateau procedure."""

    selected_scale: float
    selected_dimension: int | float
    plateau_selection: PlateauSelection
    method: str = "plateau"


@dataclass(frozen=True)
class PlateauCandidate:
    """One bounded dimension plateau, scored by its absolute log-width."""

    dimension: int | float
    left: float
    right: float
    center: float
    log_width: float
    persistence_score: float


@dataclass(frozen=True)
class PlateauSelection:
    """Result of comparing absolute log-widths of bounded plateaus.

    ``persistence_score`` is the absolute log-width, ``runner_up_score`` is
    the next largest log-width, and ``score_margin`` is their difference.
    """

    succeeded: bool
    dimension: int | float | None
    left: float | None
    right: float | None
    center: float | None
    log_width: float | None
    persistence_score: float | None
    runner_up_score: float | None
    score_margin: float | None
    failure_reason: str | None


@dataclass(frozen=True)
class _Line:
    slope: float
    intercept: float
    dimension: int | float


def _finite_nonnegative_float(value, name: str) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite, nonnegative number.") from exc
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite, nonnegative number.")
    return value


def _validate_inputs(d_m_values, objective_values, penalty_values,
                     *, require_monotonic_penalty=True):
    dimensions_original = np.asarray(d_m_values)
    try:
        dimensions = np.asarray(d_m_values, dtype=float)
        objectives = np.asarray(objective_values, dtype=float)
        penalties = np.asarray(penalty_values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "d_m_values, objective_values, and penalty_values must be numeric."
        ) from exc

    arrays = {
        "d_m_values": dimensions,
        "objective_values": objectives,
        "penalty_values": penalties,
    }
    for name, values in arrays.items():
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")

    if len(dimensions) == 0:
        raise ValueError("At least one candidate dimension is required.")
    if not (len(dimensions) == len(objectives) == len(penalties)):
        raise ValueError(
            "d_m_values, objective_values, and penalty_values must have "
            "the same length."
        )
    if not np.all(np.isfinite(dimensions)) or np.any(dimensions < 0.0):
        raise ValueError("d_m_values must contain finite, nonnegative values.")
    if len(np.unique(dimensions)) != len(dimensions):
        raise ValueError("d_m_values must contain unique candidate dimensions.")
    if not np.all(np.isfinite(penalties)) or np.any(penalties < 0.0):
        raise ValueError("penalty_values must contain finite, nonnegative values.")

    finite_objectives = np.isfinite(objectives)
    if not np.any(finite_objectives):
        raise ValueError("At least one objective value must be finite.")

    dimension_order = np.argsort(dimensions)
    if require_monotonic_penalty and np.any(np.diff(penalties[dimension_order]) < 0.0):
        raise ValueError(
            "penalty_values must be nondecreasing with model dimension."
        )

    return (
        dimensions_original,
        dimensions,
        objectives,
        penalties,
        finite_objectives,
    )


def _intersection(left: _Line, right: _Line) -> float:
    """Return where a lower-slope right line overtakes the left line."""

    return (right.intercept - left.intercept) / (left.slope - right.slope)


def build_dimension_path(
    d_m_values,
    objective_values,
    penalty_values,
    *,
    require_monotonic_penalty=True,
) -> DimensionPath:
    """Build the exact lower-envelope path of selected dimensions.

    Each candidate model defines the affine criterion

    ``objective_values[m] + C * penalty_values[m]``.

    The returned path is exact up to floating-point arithmetic and therefore
    does not depend on a user-chosen grid of ``C`` values.  Non-finite
    objective values are excluded. By default penalties must be
    nondecreasing with model dimension, as required by bootstrap screening.
    Plain Plateau can disable this check for dimension-dependent penalties.
    """

    (
        dimensions_original,
        dimensions,
        objectives,
        penalties,
        finite_objectives,
    ) = _validate_inputs(d_m_values, objective_values, penalty_values,
                         require_monotonic_penalty=require_monotonic_penalty)

    eligible_indices = np.flatnonzero(finite_objectives)
    # For identical penalty slopes, only the smallest-intercept line can be
    # selected.  If the criteria are identical, retain the smaller dimension,
    # preserving the smallest-dimension tie convention.
    lines_by_slope: dict[float, _Line] = {}
    for index in eligible_indices:
        dimension_value = dimensions_original[index]
        if getattr(dimension_value, "shape", None) == ():
            dimension_value = dimension_value.item()
        line = _Line(
            slope=float(penalties[index]),
            intercept=float(objectives[index]),
            dimension=dimension_value,
        )
        previous = lines_by_slope.get(line.slope)
        if previous is None or line.intercept < previous.intercept:
            lines_by_slope[line.slope] = line
        elif (
            line.intercept == previous.intercept
            and float(line.dimension) < float(previous.dimension)
        ):
            lines_by_slope[line.slope] = line

    # With slopes in decreasing order, each accepted line becomes optimal to
    # the right of its start value.  Popping non-increasing starts constructs
    # the lower envelope in linear time after sorting.
    ordered_lines = sorted(
        lines_by_slope.values(),
        key=lambda line: line.slope,
        reverse=True,
    )
    envelope: list[_Line] = []
    starts: list[float] = []
    for line in ordered_lines:
        start = float("-inf")
        while envelope:
            start = _intersection(envelope[-1], line)
            if start > starts[-1]:
                break
            envelope.pop()
            starts.pop()
        if not envelope:
            start = float("-inf")
        envelope.append(line)
        starts.append(start)

    active_index = bisect_right(starts, 0.0) - 1
    breakpoints = [0.0]
    selected_dimensions = [envelope[active_index].dimension]
    for index in range(active_index + 1, len(envelope)):
        start = float(starts[index])
        if not np.isfinite(start):
            continue
        if start <= 0.0:
            selected_dimensions[0] = envelope[index].dimension
            continue
        if envelope[index].dimension == selected_dimensions[-1]:
            continue
        breakpoints.append(start)
        selected_dimensions.append(envelope[index].dimension)

    return DimensionPath(
        breakpoints=tuple(breakpoints),
        dimensions=tuple(selected_dimensions),
    )


def ranked_plateaus(path: DimensionPath) -> list[PlateauCandidate]:
    """Rank bounded plateaus by log-width, breaking exact ties by smaller D."""
    candidates: list[PlateauCandidate] = []
    for path_index in range(1, len(path.dimensions) - 1):
        left = float(path.breakpoints[path_index])
        right = float(path.breakpoints[path_index + 1])
        log_width = log(right) - log(left)
        candidates.append(
            PlateauCandidate(
                dimension=path.dimensions[path_index],
                left=left,
                right=right,
                center=sqrt(left * right),
                log_width=log_width,
                persistence_score=log_width,
            )
        )

    return sorted(candidates, key=lambda candidate: (-candidate.log_width, candidate.dimension))


def select_persistent_plateau(path: DimensionPath) -> PlateauSelection:
    """Select the widest bounded log plateau; tied maxima remain ambiguous."""
    candidates = ranked_plateaus(path)
    if not candidates:
        return PlateauSelection(
            succeeded=False,
            dimension=None,
            left=None,
            right=None,
            center=None,
            log_width=None,
            persistence_score=None,
            runner_up_score=None,
            score_margin=None,
            failure_reason=(
                "At least one bounded plateau is required for plateau selection."
            ),
        )

    largest_score = max(candidate.persistence_score for candidate in candidates)
    tied_candidates = [
        candidate
        for candidate in candidates
        if np.isclose(
            candidate.persistence_score,
            largest_score,
            rtol=1e-12,
            atol=1e-12,
        )
    ]
    if len(tied_candidates) > 1:
        return PlateauSelection(
            succeeded=False,
            dimension=None,
            left=None,
            right=None,
            center=None,
            log_width=None,
            persistence_score=float(largest_score),
            runner_up_score=float(largest_score),
            score_margin=0.0,
            failure_reason=(
                "Plateau comparison is ambiguous because multiple plateaus "
                "have the same largest absolute log-width."
            ),
        )

    winner = tied_candidates[0]
    other_scores = [
        candidate.persistence_score
        for candidate in candidates
        if candidate is not winner
    ]
    runner_up_score = max(other_scores) if other_scores else None
    score_margin = (
        winner.persistence_score - runner_up_score
        if runner_up_score is not None
        else None
    )
    return PlateauSelection(
        succeeded=True,
        dimension=winner.dimension,
        left=winner.left,
        right=winner.right,
        center=winner.center,
        log_width=winner.log_width,
        persistence_score=winner.persistence_score,
        runner_up_score=(
            float(runner_up_score) if runner_up_score is not None else None
        ),
        score_margin=float(score_margin) if score_margin is not None else None,
        failure_reason=None,
    )


def select_plateau(
    d_m_values,
    objective_values,
    penalty_values,
    *,
    require_monotonic_penalty: bool = True,
) -> PlateauScaleSelection:
    """Select from the widest bounded plateau of the penalized dimension path.

    Select at the geometric center of that plateau. Missing bounded plateaus
    and tied maximum log-widths raise explicit errors.
    """
    path = build_dimension_path(
        d_m_values, objective_values, penalty_values,
        require_monotonic_penalty=require_monotonic_penalty,
    )
    plateau = select_persistent_plateau(path)
    if not plateau.succeeded:
        raise ValueError(f"Plateau selection failed: {plateau.failure_reason}")

    selected_scale = float(plateau.center)
    return PlateauScaleSelection(
        selected_scale=selected_scale,
        selected_dimension=path.dimension_at(selected_scale),
        plateau_selection=plateau,
    )


@dataclass(frozen=True)
class BootstrapComparison:
    smaller_dimension: int
    larger_dimension: int
    observed_gain: float
    exceedances: int
    p_value: float
    retain_larger: bool
    bootstrap_gains: tuple[float, ...]
    null_spectral_radius: float


@dataclass(frozen=True)
class BootstrapSelection:
    selected_dimension: int
    candidate_dimensions: tuple[int, ...]
    plateaus: tuple
    comparisons: tuple[BootstrapComparison, ...]
    selected_edges: tuple[tuple[int, int], ...]
    top_plateaus: int
    bootstrap_replicates: int
    alpha: float
    seed: int
    num_samples: int
    fit_settings: FitSettings
    precision: float | None
    method: str = "plateau-bootstrap"


def _fit(Sigma, mask, settings, *, bootstrap=False):
    options = asdict(settings)
    # Fixed omega is part of the generating model. Sampling may put the
    # empirical minimum eigenvalue below it; do not discard those draws.
    if not bootstrap and settings.omega_fixed > float(np.linalg.eigvalsh(Sigma)[0]):
        raise ValueError("omega_ref cannot exceed the smallest eigenvalue of the observed covariance.")
    result = solve_support_with_restarts(Sigma, mask, **options)
    if result is None:
        raise ValueError("No finite feasible support refit; bootstrap selection aborted.")
    return result


def _bootstrap_gain(task):
    Sigma, smaller, larger, settings = task
    small_fit = _fit(Sigma, smaller, settings, bootstrap=True)
    large_fit = _fit(Sigma, larger, settings, bootstrap=True)
    # The smaller fit is also feasible on the larger mask. This protects
    # against a worse local optimizer solution without changing supports.
    return float(max(0.0, small_fit[2] - large_fit[2]))


def _null_covariance(fit):
    Lambda, omega, _ = fit
    radius = float(np.max(np.abs(np.linalg.eigvals(Lambda))))
    if not np.isfinite(radius) or radius >= 1 or not np.isfinite(omega) or omega <= 0:
        raise ValueError("The smaller fit must have positive omega and spectral radius < 1.")
    Sigma = solve_discrete_lyapunov(Lambda.T, omega * np.eye(len(Lambda)))
    Sigma = (Sigma + Sigma.T) / 2
    if not np.all(np.isfinite(Sigma)):
        raise ValueError("The smaller fit has no finite stationary covariance.")
    np.linalg.cholesky(Sigma)
    return Sigma, radius


def select_plateau_bootstrap(
    dimensions, objectives, penalties, raw_objectives, Sigma, support_masks,
    support_valid, num_samples, *, top_plateaus=3, bootstrap_replicates=199,
    alpha=0.05, seed=20260913, n_jobs=1, fit_settings=None,
    true_support=None, progress=None,
):
    """Screen top log-width plateaus, then test adjacent descending dimensions.

    Screening uses the supplied (possibly floored) objectives. Tests use raw
    objectives, refitting both original masks for every draw. Gaussian samples
    have known zero mean: Wishart(N, Sigma)/N is exactly X.T @ X/N. For N < n,
    draw X explicitly because the Wishart sampler requires N >= n.
    """
    for name, value in (("top_plateaus", top_plateaus),
                        ("bootstrap_replicates", bootstrap_replicates),
                        ("num_samples", num_samples), ("n_jobs", n_jobs)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer.")
    settings = fit_settings or FitSettings()
    if settings.init_strategy != "halton":
        raise ValueError("Bootstrap refitting currently requires deterministic Halton starts.")
    if settings.max_restarts < 1 or settings.max_iter < 1 or settings.beta <= 0:
        raise ValueError("Fit iteration/restart counts and beta must be positive.")
    Sigma = np.asarray(Sigma, dtype=float)
    if Sigma.ndim != 2 or Sigma.shape[0] != Sigma.shape[1] or not np.all(np.isfinite(Sigma)):
        raise ValueError("Sigma must be a finite square matrix.")
    if not np.allclose(Sigma, Sigma.T):
        raise ValueError("Sigma must be symmetric.")
    n = len(Sigma)
    dims = np.asarray(dimensions)
    masks = np.asarray(support_masks, dtype=bool)
    valid = np.asarray(support_valid, dtype=bool)
    raw = np.asarray(raw_objectives, dtype=float)
    if masks.shape != (len(dims), n, n) or valid.shape != dims.shape or raw.shape != dims.shape:
        raise ValueError("Support masks, validity flags and raw objectives must match dimensions.")
    path = build_dimension_path(dims, objectives, penalties)
    plateaus = tuple(ranked_plateaus(path)[:top_plateaus])
    if not plateaus:
        raise ValueError("At least one bounded plateau is required for bootstrap selection.")
    candidates = tuple(sorted((int(p.dimension) for p in plateaus), reverse=True))
    indices = {int(d): i for i, d in enumerate(dims)}
    for d in candidates:
        i = indices[d]
        if not valid[i] or not np.isfinite(raw[i]):
            raise ValueError(f"Missing valid support/objective at dimension {d}.")
        validate_support_mask(masks[i], n, d - 1, off_diagonal_edges(n))
    for larger, smaller in zip(candidates, candidates[1:]):
        if np.any(masks[indices[smaller]] & ~masks[indices[larger]]):
            raise ValueError("Plateau candidates must be nested; recompute with nested_supports=True.")

    fits = {}
    comparisons = []
    selected = candidates[-1]
    # One stream per dimension pair: reproducible independent of worker count,
    # and the same pair uses the same draws when top_plateaus changes.
    for larger, smaller in zip(candidates, candidates[1:]):
        if progress:
            progress(f"Bootstrap comparison D_m={larger} versus {smaller} ({bootstrap_replicates} replicates)")
        for d in (smaller, larger):
            if d not in fits:
                fits[d] = _fit(Sigma, masks[indices[d]], settings)
                if not np.isclose(fits[d][2], raw[indices[d]], rtol=1e-5, atol=1e-10):
                    raise ValueError(
                        f"Refitted objective at D_m={d} differs from the saved curve; "
                        "use matching fitting settings or regenerate the curve."
                    )
        observed = float(max(0.0, raw[indices[smaller]] - raw[indices[larger]]))
        null_sigma, radius = _null_covariance(fits[smaller])
        rng = np.random.default_rng(np.random.SeedSequence([seed, smaller, larger]))
        if num_samples >= n:
            samples = np.asarray(wishart.rvs(
                df=num_samples, scale=null_sigma, size=bootstrap_replicates, random_state=rng,
            )).reshape(bootstrap_replicates, n, n) / num_samples
        else:
            x = rng.multivariate_normal(np.zeros(n), null_sigma, size=(bootstrap_replicates, num_samples))
            samples = x.swapaxes(-1, -2) @ x / num_samples
        tasks = [(s, masks[indices[smaller]], masks[indices[larger]], settings) for s in samples]
        if n_jobs == 1:
            gains = tuple(map(_bootstrap_gain, tasks))
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                gains = tuple(executor.map(_bootstrap_gain, tasks))
        exceedances = int(np.count_nonzero(np.asarray(gains) >= observed))
        p_value = (1 + exceedances) / (bootstrap_replicates + 1)
        retain = p_value < alpha
        comparisons.append(BootstrapComparison(
            smaller, larger, observed, exceedances, p_value, retain, gains, radius,
        ))
        if progress:
            progress(f"p={p_value:.6g}: {'retain larger' if retain else 'prefer smaller'}")
        if retain:
            selected = larger
            break

    selected_mask = masks[indices[selected]].copy()
    np.fill_diagonal(selected_mask, False)
    edges = tuple((int(i), int(j)) for i, j in np.argwhere(selected_mask))
    precision = None
    if true_support is not None:
        truth = np.asarray(true_support, dtype=bool)
        if truth.shape != (n, n):
            raise ValueError("true_support must match Sigma's shape.")
        precision = float(np.count_nonzero(truth & selected_mask) / len(edges)) if edges else None
    return BootstrapSelection(
        selected, candidates, plateaus, tuple(comparisons), edges, top_plateaus,
        bootstrap_replicates, float(alpha), int(seed), int(num_samples), settings, precision,
    )
