"""Construct objective/support curves using the MS-S forward-nested search."""
from __future__ import annotations

import numpy as np

from .config import FitSettings, validate_covariance, positive_integer, nonnegative_float
from .optimizers.support_search import optimize_lambda, resolve_omega_ref
from .supports.common import off_diagonal_edges, upper_triangular_edges, validate_support_mask
from .results import SupportCurve

def compute_objective_curve(
    Sigma,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    zero_tol=1e-5,
    obj_tol=1e-8,
    max_restarts=5,
    fallback_candidates=500,
    fallback_max_fro_norm=0.95,
    random_seed=42,
    fallback_seed=None,
    omega_ref=None,
    stop_obj_threshold=1e-6,
    n_jobs=1,
    init_strategy="halton",
    return_metadata=False,
    initial_curve_result=None,
    save_callback=None,
    support_scope="all",
    nested_supports=True,
    fit_omega_ref=False,
    kappa=0.93,
    progress=None,
    return_fits=False,
):
    """Fit each dimension, by default along a greedy forward-nested path."""
    def _report(message):
        if progress is not None:
            progress(message)

    if return_fits and initial_curve_result is not None:
        raise ValueError("Returning fitted parameters is not supported for legacy resumed curves.")
    fitted_lambdas, fitted_omegas = [], []
    omega_ref = resolve_omega_ref(Sigma, omega_ref, fit_omega_ref, kappa)
    n = Sigma.shape[0]
    if support_scope not in ("all", "upper"):
        raise ValueError("support_scope must be one of ('all', 'upper').")
    allowed_edges = (
        upper_triangular_edges(n) if support_scope == "upper" else off_diagonal_edges(n)
    )
    max_dm = len(allowed_edges) + 1

    if initial_curve_result is None:
        exact_d_m_values = []
        exact_objective_values = []
        fallback_d_m_values = []
        fallback_objective_values = []
        selected_support_masks = []
        selected_support_valid = []
    else:
        if len(initial_curve_result) == 4:
            (
                initial_d_m_values,
                initial_objective_values,
                initial_fallback_d_m_values,
                initial_fallback_objective_values,
            ) = initial_curve_result
            initial_selected_support_masks = np.zeros(
                (len(initial_d_m_values), n, n),
                dtype=bool,
            )
            initial_selected_support_valid = np.zeros(
                len(initial_d_m_values),
                dtype=bool,
            )
        else:
            (
                initial_d_m_values,
                initial_objective_values,
                initial_fallback_d_m_values,
                initial_fallback_objective_values,
                initial_selected_support_masks,
                initial_selected_support_valid,
            ) = initial_curve_result
        exact_d_m_values = list(initial_d_m_values)
        exact_objective_values = list(initial_objective_values)
        fallback_d_m_values = list(initial_fallback_d_m_values)
        fallback_objective_values = list(initial_fallback_objective_values)
        selected_support_masks = [
            np.array(mask, dtype=bool)
            for mask in initial_selected_support_masks
        ]
        selected_support_valid = [
            bool(valid)
            for valid in initial_selected_support_valid
        ]

    if nested_supports:
        count = len(exact_d_m_values)
        if count > max_dm or not np.array_equal(exact_d_m_values, np.arange(1, count + 1)):
            raise ValueError("Nested recovery requires a contiguous saved dimension prefix starting at D_m=1.")
        if not (len(exact_objective_values) == len(selected_support_masks) == len(selected_support_valid) == count):
            raise ValueError("Nested recovery requires one objective and support mask per saved dimension.")
        previous = np.eye(n, dtype=bool)
        failed = False
        for index, (mask, valid, obj) in enumerate(zip(
            selected_support_masks, selected_support_valid, exact_objective_values,
        )):
            if not valid:
                if not np.isposinf(obj):
                    raise ValueError("Nested recovery requires saved support masks for every finite objective.")
                failed = True
                continue
            if failed or not np.isfinite(obj):
                raise ValueError("A valid nested support cannot follow a failed dimension.")
            mask = validate_support_mask(mask, n, index, allowed_edges)
            if np.any(previous & ~mask):
                raise ValueError("Saved supports are not nested; use nested_supports=False for unrestricted curves.")
            previous = mask

    completed_d_m_values = set(int(d_m) for d_m in exact_d_m_values)

    for d_m in range(1, max_dm + 1):
        if d_m in completed_d_m_values:
            _report(f"Skipping D_m = {d_m}; already present in output file.")
            continue

        _report(f"Solving for D_m = {d_m}...")
        previous_mask = None
        blocked = nested_supports and d_m > 1 and not selected_support_valid[-1]
        if nested_supports and d_m > 1 and not blocked:
            previous_mask = selected_support_masks[-1]
        if blocked:
            _report("  No valid preceding support to extend.")
            solve_result = (None, None, np.inf, {})
        else:
            solve_result = optimize_lambda(
                Sigma,
                d_m,
                beta=beta,
                max_iter=max_iter,
                tol=tol,
                zero_tol=zero_tol,
                obj_tol=obj_tol,
                max_restarts=max_restarts,
                omega_ref=omega_ref,
                n_jobs=n_jobs,
                random_seed=random_seed,
                init_strategy=init_strategy,
                return_metadata=True,
                support_scope=support_scope,
                previous_support_mask=previous_mask,
            )
        Lambda, omega, obj, metadata = solve_result

        if (
            Lambda is None
            or omega is None
            or not np.all(np.isfinite(Lambda))
            or not np.isfinite(omega)
            or not np.isfinite(obj)
        ):
            _report(
                f"  No finite solution was found for D_m = {d_m}; "
                "recording objective = Inf."
            )
            exact_d_m_values.append(d_m)
            exact_objective_values.append(np.inf)
            selected_support_masks.append(np.zeros((n, n), dtype=bool))
            selected_support_valid.append(False)
            fitted_lambdas.append(np.full((n, n), np.nan))
            fitted_omegas.append(np.nan)
            completed_d_m_values.add(d_m)
            if save_callback is not None:
                save_callback(
                    (
                        np.array(exact_d_m_values),
                        np.array(exact_objective_values),
                        np.array(fallback_d_m_values),
                        np.array(fallback_objective_values),
                        np.array(selected_support_masks, dtype=bool),
                        np.array(selected_support_valid, dtype=bool),
                    )
                )
            continue

        if nested_supports:
            chosen = validate_support_mask(metadata["selected_support_mask"], n, d_m - 1, allowed_edges)
            if previous_mask is not None and np.any(previous_mask & ~chosen):
                raise ValueError("Selected support does not extend the previous dimension.")
        _report(f"  Objective = {obj:.6f}")
        exact_d_m_values.append(d_m)
        exact_objective_values.append(obj)
        selected_support_masks.append(
            np.array(metadata["selected_support_mask"], dtype=bool)
        )
        selected_support_valid.append(True)
        fitted_lambdas.append(np.array(Lambda, copy=True))
        fitted_omegas.append(float(omega))
        completed_d_m_values.add(d_m)
        if save_callback is not None:
            save_callback(
                (
                    np.array(exact_d_m_values),
                    np.array(exact_objective_values),
                    np.array(fallback_d_m_values),
                    np.array(fallback_objective_values),
                    np.array(selected_support_masks, dtype=bool),
                    np.array(selected_support_valid, dtype=bool),
                )
            )

    d_m_values = np.array(exact_d_m_values)
    objective_values = np.array(exact_objective_values)

    result = (
        d_m_values,
        objective_values,
        np.array(fallback_d_m_values),
        np.array(fallback_objective_values),
        np.array(selected_support_masks, dtype=bool),
        np.array(selected_support_valid, dtype=bool),
    )
    if return_fits:
        return result + (np.asarray(fitted_lambdas), np.asarray(fitted_omegas))
    return result


def compute_support_curve(
    sigma_hat, *, num_samples=None, max_restarts=10, omega_star=None,
    omega_ref=None, fit_omega_ref=True, kappa=0.93, support_scope="all",
    nested_supports=True,
    beta=1.0, max_iter=800, tol=1e-7, zero_tol=1e-5, obj_tol=1e-8,
    init_strategy="halton", random_seed=42, n_jobs=1,
    progress=None,
):
    """Fit an in-memory curve using greedy one-edge extensions by default.

    The default reference is kappa * lambda_min(sigma_hat), held fixed. A known
    numeric reference requires fit_omega_ref=False. Disabling estimation without
    supplying omega_ref is an error. omega_star records optional known
    population noise and never silently changes omega_ref or generates samples.
    No files are read/written, and progress is silent unless a callback is given.
    """
    sigma = validate_covariance(sigma_hat)
    if num_samples is not None:
        num_samples = positive_integer(num_samples, "num_samples", minimum=2)
    n_jobs = positive_integer(n_jobs, "n_jobs")
    random_seed = positive_integer(random_seed, "random_seed", minimum=0)
    obj_tol = nonnegative_float(obj_tol, "obj_tol")
    for name, value in (("fit_omega_ref", fit_omega_ref), ("nested_supports", nested_supports)):
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be a callable or None.")
    if omega_star is not None:
        omega_star = nonnegative_float(omega_star, "omega_star")
    if omega_ref is not None:
        omega_ref = nonnegative_float(omega_ref, "omega_ref")
    resolved = resolve_omega_ref(sigma, omega_ref, fit_omega_ref, kappa)
    if resolved > np.linalg.eigvalsh(sigma)[0]:
        raise ValueError("omega_ref cannot exceed the smallest eigenvalue of sigma_hat for the observed fit.")
    settings = FitSettings(
        omega_fixed=resolved, beta=beta, max_iter=max_iter, tol=tol,
        zero_tol=zero_tol, max_restarts=max_restarts,
        init_strategy=init_strategy,
    )
    if support_scope not in ("all", "upper"):
        raise ValueError("support_scope must be 'all' or 'upper'.")
    allowed = tuple(
        upper_triangular_edges(len(sigma)) if support_scope == "upper"
        else off_diagonal_edges(len(sigma))
    )
    raw = compute_objective_curve(
        sigma, beta=settings.beta, max_iter=settings.max_iter, tol=settings.tol,
        zero_tol=settings.zero_tol, obj_tol=obj_tol, max_restarts=settings.max_restarts,
        omega_ref=resolved,
        n_jobs=n_jobs, random_seed=random_seed, init_strategy=settings.init_strategy,
        support_scope=support_scope,
        nested_supports=nested_supports, progress=progress, return_fits=True,
    )
    return SupportCurve(
        sigma_hat=sigma, dimensions=raw[0], raw_objectives=raw[1],
        support_masks=raw[4], support_valid=raw[5], fitted_lambdas=raw[6],
        fitted_omegas=raw[7], fit_settings=settings, resolved_omega_ref=resolved,
        allowed_edges=allowed, support_scope=support_scope,
        nested_supports=bool(nested_supports), num_samples=num_samples,
        omega_star=omega_star,
    )
