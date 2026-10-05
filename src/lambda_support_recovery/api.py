"""Public covariance-to-support API and selection from cached curves."""
from __future__ import annotations

from math import comb

import numpy as np

from .config import PenaltyConfig, build_penalty_constants, nonnegative_float, positive_integer
from .curve import compute_support_curve
from .penalty import compute_K, compute_v
from .results import SelectionResult, SupportCurve
from .selection import select_plateau, select_plateau_bootstrap


def floor_objective_values(objective_values, objective_floor=1e-8):
    """Tie finite objectives <= the floor at zero, preserving Inf and raw data."""
    try:
        objective_floor = nonnegative_float(objective_floor, "objective_floor")
    except ValueError as exc:
        raise ValueError("objective_floor must be a finite, nonnegative number.") from exc
    values = np.array(objective_values, dtype=float, copy=True)
    floored = np.isfinite(values) & (values <= objective_floor)
    values[floored] = 0.0
    return values, floored


def _selection_options(method, lm_mode, lm_weight, objective_floor,
                       top_plateaus, bootstrap_replicates, bootstrap_alpha,
                       bootstrap_seed, n_jobs, recommendation_factor):
    if not isinstance(method, str):
        raise ValueError("method must be 'plateau-bootstrap' or 'plateau'.")
    method = method.strip().lower().replace("_", "-")
    if method not in ("plateau-bootstrap", "plateau"):
        raise ValueError("method must be 'plateau-bootstrap' or 'plateau'.")
    if lm_mode is None:
        lm_mode = "constant" if method == "plateau-bootstrap" else "support-count"
    if lm_mode not in ("constant", "support-count"):
        raise ValueError("lm_mode must be 'constant' or 'support-count'.")
    if method == "plateau-bootstrap" and lm_mode != "constant":
        raise ValueError("Plateau_Bootstrap currently requires lm_mode='constant'; use plain Plateau for support-count weighting.")
    weight = 1.0 if lm_mode == "constant" else 0.1
    weight = nonnegative_float(weight if lm_weight is None else lm_weight, "lm_weight", positive=True)
    objective_floor = nonnegative_float(objective_floor, "objective_floor")
    positive_integer(n_jobs, "n_jobs")
    nonnegative_float(recommendation_factor, "recommendation_factor", positive=True)
    if method == "plateau-bootstrap":
        positive_integer(top_plateaus, "top_plateaus")
        positive_integer(bootstrap_replicates, "bootstrap_replicates")
        positive_integer(bootstrap_seed, "bootstrap_seed", minimum=0)
        alpha = nonnegative_float(bootstrap_alpha, "bootstrap_alpha", positive=True)
        if alpha >= 1:
            raise ValueError("bootstrap_alpha must be between 0 and 1.")
    return method, lm_mode, weight, objective_floor


def select_from_curve(
    curve, *, num_samples=None, method="plateau-bootstrap",
    objective_floor=1e-8, lm_weight=None, lm_mode=None,
    penalty_config=None, top_plateaus=3, bootstrap_replicates=199,
    bootstrap_alpha=0.05, bootstrap_seed=20260913, n_jobs=1,
    recommendation_factor=2.0, return_result=False, progress=None,
):
    """Select from a stored curve without repeating support reconstruction.

    Bootstrap still refits the screened fixed masks on simulated covariances.
    The recorded original sample size cannot be replaced by a different N.
    Plain Plateau uses support-count weighting by default and selects at twice
    the geometric center of its widest bounded plateau, preserving MS-S.
    """
    if not isinstance(curve, SupportCurve):
        raise TypeError("curve must be a SupportCurve returned by compute_support_curve.")
    if num_samples is None:
        num_samples = curve.num_samples
    num_samples = positive_integer(num_samples, "num_samples", minimum=2)
    if curve.num_samples is not None and num_samples != curve.num_samples:
        raise ValueError("num_samples must match the original sample size recorded in the curve.")
    method, lm_mode, weight, objective_floor = _selection_options(
        method, lm_mode, lm_weight, objective_floor, top_plateaus,
        bootstrap_replicates, bootstrap_alpha, bootstrap_seed, n_jobs,
        recommendation_factor,
    )
    if not isinstance(return_result, (bool, np.bool_)):
        raise ValueError("return_result must be a boolean.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be a callable or None.")
    if method == "plateau-bootstrap":
        if not curve.nested_supports:
            raise ValueError("Plateau_Bootstrap requires nested_supports=True.")
        if curve.fit_settings.init_strategy != "halton":
            raise ValueError("Plateau_Bootstrap requires init_strategy='halton'.")
    constants = build_penalty_constants(curve.sigma_hat, num_samples, config=penalty_config, lm_weight=weight)
    if lm_mode == "constant":
        lm_values = np.full(len(curve.dimensions), weight)
    else:
        counts = [comb(len(curve.allowed_edges), int(dimension) - 1) for dimension in curve.dimensions]
        try:
            lm_values = weight * np.asarray(counts, dtype=float)
        except OverflowError as exc:
            raise ValueError("Support-count weights exceed floating-point range.") from exc
    # K and v are independent of dimension; evaluate the entropy integral once.
    K, v = compute_K(constants), compute_v(constants)
    with np.errstate(over="ignore", invalid="ignore"):
        penalties = np.sqrt(curve.dimensions) * (K + np.sqrt(2 * v * lm_values))
    if not np.all(np.isfinite(penalties)):
        raise ValueError("Penalty quantities exceed floating-point range; use smaller reference bounds.")
    screened, _ = floor_objective_values(curve.raw_objectives, objective_floor)
    if method == "plateau-bootstrap":
        diagnostics = select_plateau_bootstrap(
            curve.dimensions, screened, penalties, curve.raw_objectives,
            curve.sigma_hat, curve.support_masks, curve.support_valid,
            num_samples, top_plateaus=top_plateaus,
            bootstrap_replicates=bootstrap_replicates, alpha=bootstrap_alpha,
            seed=bootstrap_seed, n_jobs=n_jobs, fit_settings=curve.fit_settings,
            progress=progress,
        )
    else:
        diagnostics = select_plateau(
            curve.dimensions, screened, penalties,
            recommendation_factor=recommendation_factor,
            require_monotonic_penalty=lm_mode == "constant",
        )
    selected = int(diagnostics.selected_dimension)
    index = int(np.flatnonzero(curve.dimensions == selected)[0])
    if not curve.support_valid[index]:
        raise ValueError("Selected dimension does not have a valid fitted support.")
    mask = curve.support_masks[index].copy()
    edges = tuple((int(i), int(j)) for i, j in np.argwhere(mask & ~np.eye(len(mask), dtype=bool)))
    result = SelectionResult(
        support=mask, selected_dimension=selected, selected_edges=edges,
        method=method, curve=curve, num_samples=num_samples,
        penalty_constants=constants, penalty_values=penalties,
        lm_values=lm_values, lm_mode=lm_mode, objective_floor=objective_floor,
        screened_objectives=screened, diagnostics=diagnostics,
    )
    return result if return_result else mask


def select_support(
    sigma_hat, *, num_samples, method="plateau-bootstrap", max_restarts=10,
    omega_star=None, omega_ref=None, fit_omega_ref=True,
    objective_floor=1e-8, lm_weight=None, lm_mode=None, penalty_config=None,
    top_plateaus=3, bootstrap_replicates=199, bootstrap_alpha=0.05,
    kappa=0.93, support_scope="all", nested_supports=True,
    preselect_edges=None, refine_after_fixed_omega=False,
    beta=1.0, max_iter=800, tol=1e-7, zero_tol=1e-5, obj_tol=1e-8,
    min_omega=0.0, init_strategy="halton", random_seed=42,
    bootstrap_seed=20260913, n_jobs=1, recommendation_factor=2.0,
    return_result=False, progress=None,
):
    """Select the Lambda model support from an empirical covariance and N.

    Defaults: estimated fixed reference omega, greedy nested support curve,
    and Plateau_Bootstrap. Return an n-by-n Boolean mask including diagonals,
    or a SelectionResult when return_result=True. PenaltyConfig controls the
    penalty covariance reference (observed covariance by default), Lambda bound
    (1 by default), noise bound, confidence constant, and individual summaries.

    Plain Plateau defaults to support-count Lm = 0.1 * C(M, D_m - 1). Constant
    Lm=1 is used for bootstrap screening. Constant weights rescale the penalty
    axis without changing plateau widths. omega_star is optional provenance;
    use omega_ref=known_omega and fit_omega_ref=False to fit with known noise.

    The bootstrap assumes iid zero-mean Gaussian samples with Sigma_hat=X.T@X/N.
    Failed fits or absent/ambiguous plateaus raise ValueError, with no automatic
    alternative selection procedure. No filesystem or plotting side effects.
    """
    num_samples = positive_integer(num_samples, "num_samples", minimum=2)
    normalized, resolved_mode, weight, objective_floor = _selection_options(
        method, lm_mode, lm_weight, objective_floor, top_plateaus,
        bootstrap_replicates, bootstrap_alpha, bootstrap_seed, n_jobs,
        recommendation_factor,
    )
    if not isinstance(return_result, (bool, np.bool_)):
        raise ValueError("return_result must be a boolean.")
    if normalized == "plateau-bootstrap" and (not nested_supports or init_strategy != "halton"):
        raise ValueError("Plateau_Bootstrap requires nested_supports=True and init_strategy='halton'.")
    # Reject invalid penalty configuration before the expensive support search.
    build_penalty_constants(sigma_hat, num_samples, config=penalty_config, lm_weight=weight)
    curve = compute_support_curve(
        sigma_hat, num_samples=num_samples, max_restarts=max_restarts,
        omega_star=omega_star, omega_ref=omega_ref, fit_omega_ref=fit_omega_ref,
        kappa=kappa, support_scope=support_scope, nested_supports=nested_supports,
        preselect_edges=preselect_edges, refine_after_fixed_omega=refine_after_fixed_omega,
        beta=beta, max_iter=max_iter, tol=tol, zero_tol=zero_tol, obj_tol=obj_tol,
        min_omega=min_omega, init_strategy=init_strategy, random_seed=random_seed,
        n_jobs=n_jobs, progress=progress,
    )
    return select_from_curve(
        curve, method=normalized, objective_floor=objective_floor,
        lm_weight=weight, lm_mode=resolved_mode, penalty_config=penalty_config,
        top_plateaus=top_plateaus, bootstrap_replicates=bootstrap_replicates,
        bootstrap_alpha=bootstrap_alpha, bootstrap_seed=bootstrap_seed,
        n_jobs=n_jobs, recommendation_factor=recommendation_factor,
        return_result=return_result, progress=progress,
    )
