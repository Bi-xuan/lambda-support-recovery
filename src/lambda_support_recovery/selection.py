"""Data-driven selection of the multiplier in a model-selection penalty.

The window and plateau procedures estimate the minimal-penalty scale from
the exact, piecewise-constant path of selected model dimensions. The slope
heuristic then recommends a rescaled value for final model selection.
Plateau-bootstrap selection directly compares the top plateau models using
fixed-support bootstrap tests.
"""

from __future__ import annotations

from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from math import expm1, log, log1p, sqrt
from typing import Mapping

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import wishart

from .optimizers.support_search import solve_support_with_restarts
from .supports.common import off_diagonal_edges, validate_support_mask


from .config import FitSettings


DEFAULT_RECOMMENDATION_FACTOR = 2.0
DEFAULT_WINDOW_PERTURBATION = 0.2
DEFAULT_WINDOW_AGGREGATION_RATIO = 1.5
DEFAULT_WINDOW_SEPARATION_RATIO = 2.5


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
class ScalingSelection:
    """Result of a minimal-penalty scale-selection procedure."""

    method: str
    minimal_scale: float
    recommended_scale: float
    selected_dimension: int | float
    eta: float | None = None
    largest_jump: float | None = None
    recommendation_factor: float = DEFAULT_RECOMMENDATION_FACTOR
    selection_source: str | None = None
    jump_selection: AdaptiveWindowSelection | None = None
    plateau_selection: PlateauSelection | None = None
    recommendation_within_plateau: bool | None = None


@dataclass(frozen=True)
class AdaptiveWindowSelection:
    """Adaptive window bandwidth and its dominant jump cluster."""

    succeeded: bool
    center: float | None
    eta: float | None
    largest_jump: float | None
    transition_scales: tuple[float, ...]
    failure_reason: str | None
    rejection_counts: Mapping[str, int]


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
class _DominantCluster:
    first: int
    last: int
    strength: float
    center: float
    num_tied: int


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


def _positive_float(value, name: str) -> float:
    value = _finite_nonnegative_float(value, name)
    if value == 0.0:
        raise ValueError(f"{name} must be positive.")
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
    nondecreasing with model dimension so that the selected-complexity path
    has the interpretation required by the jump procedures. Plateau screening
    can disable this check for dimension-dependent penalties.
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
    # matching the convention used by model_selection.select_dimension.
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


def window(path: DimensionPath, eta: float) -> float:
    """Return the geometric center of the last maximal fixed-width interval.

    This is the standalone fixed-bandwidth window procedure: ``eta`` must
    already be known, and the function returns only the selected center.
    It is not called by :func:`adaptive_window`.  The adaptive procedure
    searches for a suitable bandwidth and calls :func:`_dominant_cluster`
    directly because its stability, aggregation, and separation tests also
    need cluster membership, total strength, and tie information.  The two
    procedures nevertheless use the same final-maximizer tie-breaking rule.
    """

    eta = _positive_float(eta, "eta")
    if len(path.dimensions) < 2:
        raise ValueError("Window is undefined because the path has no transition.")

    factor = 1.0 + eta
    boundaries = {0.0}
    for transition in path.transition_scales:
        boundaries.add(float(transition) / factor)
        boundaries.add(float(transition) * factor)
    ordered_boundaries = sorted(boundaries)

    intervals: list[tuple[float, float, float]] = []
    for left, right in zip(ordered_boundaries, ordered_boundaries[1:]):
        if left == right:
            continue
        probe = sqrt(left * right) if left > 0.0 else right / 2.0
        jump = float(path.dimension_at(probe / factor)) - float(
            path.dimension_at(probe * factor)
        )
        intervals.append((left, right, jump))

    if not intervals:
        raise ValueError("Window is undefined because no finite interval is available.")
    largest_jump = max(interval[2] for interval in intervals)
    if largest_jump <= 0.0:
        raise ValueError("Window requires at least one downward dimension jump.")

    maximal_intervals = [
        interval
        for interval in intervals
        if np.isclose(interval[2], largest_jump, rtol=1e-12, atol=1e-12)
    ]

    # Merge adjacent maximizing intervals, then keep the final connected
    # component as prescribed in Appendix D.2.
    components: list[list[float]] = []
    for left, right, _ in maximal_intervals:
        if components and left == components[-1][1]:
            components[-1][1] = right
        else:
            components.append([left, right])
    left, right = components[-1]
    return sqrt(left * right)


def _dominant_cluster(
    log_transitions: np.ndarray,
    jumps: np.ndarray,
    log_half_width: float,
) -> _DominantCluster:
    """Return the last strongest transition cluster fitting in a log window."""

    cumulative_jumps = np.concatenate(([0.0], np.cumsum(jumps)))
    best_strength = float("-inf")
    tied_clusters: list[tuple[int, int, float]] = []
    last = -1
    width = 2.0 * log_half_width

    for first in range(len(log_transitions)):
        last = max(last, first)
        while (
            last + 1 < len(log_transitions)
            and log_transitions[last + 1] - log_transitions[first]
            <= width + 1e-14
        ):
            last += 1

        strength = float(cumulative_jumps[last + 1] - cumulative_jumps[first])
        center = float((log_transitions[first] + log_transitions[last]) / 2.0)
        if strength > best_strength and not np.isclose(
            strength,
            best_strength,
            rtol=1e-12,
            atol=1e-12,
        ):
            best_strength = strength
            tied_clusters = [(first, last, center)]
        elif np.isclose(strength, best_strength, rtol=1e-12, atol=1e-12):
            tied_clusters.append((first, last, center))

    # This matches window(): retain the final maximizing component on the
    # scale axis when several windows have the same strength.
    first, last, center = max(tied_clusters, key=lambda item: item[2])
    return _DominantCluster(
        first=first,
        last=last,
        strength=best_strength,
        center=center,
        num_tied=len(tied_clusters),
    )


def _cluster_is_separated(
    cluster: _DominantCluster,
    log_transitions: np.ndarray,
    separation_ratio: float,
) -> bool:
    """Return whether a multi-jump cluster is isolated from its neighbors."""

    internal_gaps = np.diff(
        log_transitions[cluster.first : cluster.last + 1]
    )
    if len(internal_gaps) == 0:
        return False

    external_gaps = []
    if cluster.first > 0:
        external_gaps.append(
            log_transitions[cluster.first]
            - log_transitions[cluster.first - 1]
        )
    if cluster.last + 1 < len(log_transitions):
        external_gaps.append(
            log_transitions[cluster.last + 1]
            - log_transitions[cluster.last]
        )
    if not external_gaps:
        return False

    return min(external_gaps) >= separation_ratio * float(np.max(internal_gaps))


def adaptive_window(
    path: DimensionPath,
    minimum_eta: float | None = None,
    *,
    perturbation: float = DEFAULT_WINDOW_PERTURBATION,
    aggregation_ratio: float = DEFAULT_WINDOW_AGGREGATION_RATIO,
    separation_ratio: float = DEFAULT_WINDOW_SEPARATION_RATIO,
) -> AdaptiveWindowSelection:
    """Choose the smallest stable bandwidth around a dominant jump cluster.

    The search is performed on the exact transition scales in ``log(C)``.
    It tests only bandwidths at which a new consecutive transition cluster
    can fit inside a window.  An aggregate is accepted when it remains the
    unique dominant cluster under a relative ``perturbation`` of the log
    half-width, contains meaningful mass beyond its largest member, and is
    separated from the nearest transition outside the cluster.

    If no multi-jump cluster meets those conditions, the returned result has
    ``succeeded=False`` and records why candidates were rejected.  It does not
    silently substitute a machine-precision singleton window.
    """

    if len(path.dimensions) < 2:
        raise ValueError("Window is undefined because the path has no transition.")

    perturbation = _positive_float(perturbation, "perturbation")
    if perturbation >= 1.0:
        raise ValueError("perturbation must be less than one.")
    aggregation_ratio = _positive_float(
        aggregation_ratio,
        "aggregation_ratio",
    )
    separation_ratio = _positive_float(separation_ratio, "separation_ratio")

    if minimum_eta is None:
        # The exact path has no sampling-grid resolution.  This is the
        # smallest practical multiplicative window distinguishable from one.
        minimum_eta = 8.0 * np.finfo(float).eps
    else:
        minimum_eta = _positive_float(minimum_eta, "minimum_eta")
    minimum_log_width = log1p(minimum_eta)

    transition_scales = np.asarray(path.transition_scales, dtype=float)
    log_transitions = np.log(transition_scales)
    jumps = np.asarray(path.dimensions[:-1], dtype=float) - np.asarray(
        path.dimensions[1:],
        dtype=float,
    )
    if np.any(jumps <= 0.0):
        raise ValueError(
            "Window requires every dimension-path transition to be downward."
        )

    critical_widths = {0.0}
    for first in range(len(log_transitions) - 1):
        for last in range(first + 1, len(log_transitions)):
            critical_widths.add(
                float((log_transitions[last] - log_transitions[first]) / 2.0)
            )

    # At a cluster's entry event, move far enough into its plateau that a
    # negative perturbation remains beyond the event.  Testing all pairwise
    # events also detects plateaus whose start is caused by a competing cluster.
    candidate_widths = sorted(
        {
            max(width, minimum_log_width)
            / (1.0 - perturbation)
            * (1.0 + 1e-12)
            for width in critical_widths
        }
    )

    rejection_counts = {
        "singleton": 0,
        "unstable_or_tied": 0,
        "insufficient_aggregation": 0,
        "insufficient_separation": 0,
    }

    for log_half_width in candidate_widths:
        probes = (
            (1.0 - perturbation) * log_half_width,
            log_half_width,
            (1.0 + perturbation) * log_half_width,
        )
        clusters = [
            _dominant_cluster(log_transitions, jumps, probe)
            for probe in probes
        ]
        cluster = clusters[1]
        signature = (cluster.first, cluster.last)
        if cluster.last == cluster.first:
            rejection_counts["singleton"] += 1
            continue
        if any(
            (candidate.first, candidate.last) != signature
            or candidate.num_tied != 1
            for candidate in clusters
        ):
            rejection_counts["unstable_or_tied"] += 1
            continue

        member_jumps = jumps[cluster.first : cluster.last + 1]
        if cluster.strength < aggregation_ratio * float(np.max(member_jumps)):
            rejection_counts["insufficient_aggregation"] += 1
            continue
        if not _cluster_is_separated(
            cluster,
            log_transitions,
            separation_ratio,
        ):
            rejection_counts["insufficient_separation"] += 1
            continue

        eta = expm1(log_half_width)
        return AdaptiveWindowSelection(
            succeeded=True,
            center=float(np.exp(cluster.center)),
            eta=float(eta),
            largest_jump=float(cluster.strength),
            transition_scales=tuple(
                float(value)
                for value in transition_scales[cluster.first : cluster.last + 1]
            ),
            failure_reason=None,
            rejection_counts=dict(rejection_counts),
        )

    return AdaptiveWindowSelection(
        succeeded=False,
        center=None,
        eta=None,
        largest_jump=None,
        transition_scales=(),
        failure_reason=(
            "No stable multi-transition cluster satisfied all jump criteria."
        ),
        rejection_counts=dict(rejection_counts),
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


def _normalize_method(method: str) -> str:
    if not isinstance(method, str):
        raise ValueError("method must be a string.")
    normalized = method.strip().lower().replace("-", "_")
    aliases = {
        "window": "window",
        "plateau": "plateau",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError("method must be one of: window or plateau.") from exc


def select_minimal_scale(
    d_m_values,
    objective_values,
    penalty_values,
    *,
    method: str = "window",
    eta: float | None = None,
    recommendation_factor: float = DEFAULT_RECOMMENDATION_FACTOR,
    require_monotonic_penalty: bool = True,
) -> ScalingSelection:
    """Estimate a minimal-penalty scale and report the recommended scale.

    Parameters
    ----------
    d_m_values, objective_values, penalty_values
        Candidate dimensions and the intercepts and slopes of their penalized
        criteria.
    method
        One of ``"window"`` or ``"plateau"``. The plateau method selects
        the bounded interval with the largest absolute log-width and uses
        its geometric center as the minimal scale.
    eta
        For ``window``, an optional positive lower bound for the adaptively
        selected window width. It is ignored by ``plateau``.
    recommendation_factor
        Positive multiplier applied to the estimated minimal scale. The
        default is 2, as prescribed by the slope heuristic.

    Returns
    -------
    ScalingSelection
        ``minimal_scale`` is the estimated minimal-penalty constant.
        ``recommended_scale`` is ``recommendation_factor * minimal_scale``,
        and ``selected_dimension`` is the dimension selected at that scale.
        ``selection_source`` is ``"jump"`` for a successful window selection
        and ``"plateau"`` for a successful plateau selection. A failed window
        selection raises ``ValueError``; it does not fall back to plateau
        selection.
    """

    normalized_method = _normalize_method(method)
    recommendation_factor = _positive_float(
        recommendation_factor,
        "recommendation_factor",
    )
    path = build_dimension_path(
        d_m_values,
        objective_values,
        penalty_values,
        require_monotonic_penalty=require_monotonic_penalty,
    )

    resolved_eta = None
    largest_jump = None
    selection_source = None
    jump_selection = None
    plateau_selection = None
    recommendation_within_plateau = None
    if normalized_method == "window":
        jump_selection = adaptive_window(path, minimum_eta=eta)
        if not jump_selection.succeeded:
            raise ValueError(
                f"Window selection failed: {jump_selection.failure_reason}"
            )
        selection_source = "jump"
        minimal_scale = float(jump_selection.center)
        resolved_eta = float(jump_selection.eta)
        largest_jump = float(jump_selection.largest_jump)
    else:
        plateau_selection = select_persistent_plateau(path)
        if not plateau_selection.succeeded:
            raise ValueError(
                f"Plateau selection failed: {plateau_selection.failure_reason}"
            )
        selection_source = "plateau"
        minimal_scale = float(plateau_selection.center)

    recommended_scale = recommendation_factor * minimal_scale
    if not np.isfinite(recommended_scale):
        raise ValueError("The recommended scale is not finite.")

    if selection_source == "plateau":
        recommendation_within_plateau = bool(
            plateau_selection.left
            <= recommended_scale
            < plateau_selection.right
        )

    return ScalingSelection(
        method=normalized_method,
        minimal_scale=float(minimal_scale),
        recommended_scale=float(recommended_scale),
        selected_dimension=path.dimension_at(recommended_scale),
        eta=resolved_eta,
        largest_jump=largest_jump,
        recommendation_factor=recommendation_factor,
        selection_source=selection_source,
        jump_selection=jump_selection,
        plateau_selection=plateau_selection,
        recommendation_within_plateau=recommendation_within_plateau,
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
    if settings.omega_fixed is not None:
        upper = None if bootstrap else float(np.linalg.eigvalsh(Sigma)[0])
    else:
        upper = float(np.linalg.eigvalsh(Sigma)[0]) - 1e-6
    result = solve_support_with_restarts(Sigma, mask, omega_upper=upper, **options)
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
