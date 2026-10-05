"""Search over support masks and optimize Lambda for each candidate support."""

import os
from functools import wraps
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from math import comb

import numpy as np

from ..admm import admm_solve
from ..config import nonnegative_float
from ..objective import frobenius_objective
from ..supports.common import (
    off_diagonal_edges, upper_triangular_edges,
    validate_support_mask,
)
from ..supports.exact import get_all_supports, get_upper_triangular_supports



def _preserve_random_state(function):
    """Keep legacy random starts from changing the caller's NumPy RNG state."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        state = np.random.get_state()
        try:
            return function(*args, **kwargs)
        finally:
            np.random.set_state(state)
    return wrapped


SUPPORT_SCOPES = ("all", "upper")


def resolve_omega_ref(Sigma_hat, omega_ref=None, fit_omega_ref=False, kappa=0.93):
    """Resolve omega once, before any fixed-reference Lambda fits."""
    if fit_omega_ref:
        if omega_ref is not None:
            raise ValueError("fit_omega_ref=True requires omega_ref=None.")
        if not np.isfinite(kappa) or not (0.0 < kappa <= 1.0):
            raise ValueError("kappa must be finite and in (0, 1].")
        omega_ref = float(kappa * np.linalg.eigvalsh(Sigma_hat)[0])

    if omega_ref is None:
        raise ValueError("omega_ref must be supplied when fit_omega_ref=False; omega is always fixed during fitting.")
    return nonnegative_float(omega_ref, "omega_ref")


def support_edges_from_mask(mask):
    n = mask.shape[0]
    return [
        (i, j)
        for i in range(n)
        for j in range(n)
        if i != j and mask[i, j]
    ]


def threshold_lambda(Lambda, zero_tol):
    Lambda_thr = Lambda.copy()
    Lambda_thr[np.abs(Lambda_thr) < zero_tol] = 0.0
    return Lambda_thr


def is_finite_candidate(Lambda, omega, obj):
    return (
        np.all(np.isfinite(Lambda))
        and np.isfinite(omega)
        and np.isfinite(obj)
    )


def print_optimization_result(Lambda, omega, obj):
    if Lambda is None or omega is None or not np.isfinite(obj):
        print("No candidate satisfied the hard constraints.")
        return

    print(f"Best objective: {obj:.6f}")
    print(f"Best omega:     {omega:.6f}")
    print(f"Best Lambda:\n{Lambda}")


def solve_support_with_restarts(
    Sigma,
    mask,
    beta,
    max_iter,
    tol,
    zero_tol,
    max_restarts,
    omega_fixed,
    init_strategy="halton",
):
    runs = []

    for restart_index in range(max_restarts):
        Lambda, omega = admm_solve(
            Sigma,
            mask,
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            max_restarts=1,
            omega_ref=omega_fixed,
            init_strategy=init_strategy,
            init_offset=restart_index,
        )
        Lambda_thr = threshold_lambda(Lambda, zero_tol)
        obj = frobenius_objective(Sigma, Lambda_thr, omega)

        if not is_finite_candidate(Lambda_thr, omega, obj):
            continue

        runs.append((Lambda_thr, omega, obj))

    if not runs:
        return None

    return min(runs, key=lambda run: run[2])


@_preserve_random_state
def solve_support_worker(task):
    (
        support_index,
        Sigma,
        mask,
        beta,
        max_iter,
        tol,
        zero_tol,
        max_restarts,
        omega_fixed,
        seed,
        init_strategy,
    ) = task

    if seed is not None:
        np.random.seed(seed)

    result = solve_support_with_restarts(
        Sigma,
        mask,
        beta=beta,
        max_iter=max_iter,
        tol=tol,
        zero_tol=zero_tol,
        max_restarts=max_restarts,
        omega_fixed=omega_fixed,
        init_strategy=init_strategy,
    )

    return support_index, mask, result


def iter_parallel_support_results(tasks, max_workers):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        task_iter = iter(tasks)
        pending = set()

        for _ in range(max_workers):
            try:
                pending.add(executor.submit(solve_support_worker, next(task_iter)))
            except StopIteration:
                break

        while pending:
            done, pending = wait(pending, return_when=FIRST_COMPLETED)

            for future in done:
                yield future.result()

                try:
                    pending.add(executor.submit(solve_support_worker, next(task_iter)))
                except StopIteration:
                    pass


def update_best_support(
    best_Lambda,
    best_omega,
    best_obj,
    best_mask,
    mask,
    result,
    obj_tol,
):
    if result is None:
        return best_Lambda, best_omega, best_obj, best_mask

    Lambda, omega, obj = result

    if best_Lambda is None or obj < best_obj - obj_tol:
        return Lambda.copy(), omega, obj, mask.copy()

    return best_Lambda, best_omega, best_obj, best_mask


def validate_support_scope(support_scope):
    if support_scope not in SUPPORT_SCOPES:
        raise ValueError(f"support_scope must be one of {SUPPORT_SCOPES}.")


@_preserve_random_state
def optimize_lambda(
    Sigma,
    D_m,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    zero_tol=1e-5,
    obj_tol=1e-8,
    max_restarts=3,
    omega_ref=None,
    support_iterator=None,
    n_jobs=1,
    random_seed=None,
    init_strategy="halton",
    return_metadata=False,
    support_scope="all",
    previous_support_mask=None,
    fit_omega_ref=False,
    kappa=0.93,
):
    """
    Optimize Lambda over a support iterator.

    By default this uses exact exhaustive support enumeration over all directed
    off-diagonal positions. Set support_scope="upper" to enumerate only strict
    upper-triangular positions. For larger problems, pass either a
    support_iterator(n, n_edge) callable or an iterable that yields support
    masks. With previous_support_mask, enumerate only its one-edge extensions
    within the selected scope and choose the strict minimum fitted objective.
    """
    metadata = {
        "selected_support_mask": None,
        "selected_support_edges": None,
        "support_scope": support_scope,
    }

    def result_tuple(Lambda, omega, obj):
        if return_metadata:
            return Lambda, omega, obj, metadata
        return Lambda, omega, obj

    n = Sigma.shape[0]
    n_edge = D_m - 1
    lambda_min_sigma = np.min(np.linalg.eigvalsh(Sigma))
    omega_ref = resolve_omega_ref(Sigma, omega_ref, fit_omega_ref, kappa)
    validate_support_scope(support_scope)

    if previous_support_mask is not None and support_iterator is not None:
        raise ValueError("previous_support_mask cannot be combined with support_iterator.")

    if support_scope == "upper" and support_iterator is not None:
        raise ValueError("support_scope='upper' cannot be combined with support_iterator.")

    if support_scope == "upper":
        max_upper_edges = n * (n - 1) // 2
        if not (0 <= n_edge <= max_upper_edges):
            raise ValueError(
                "For support_scope='upper', D_m - 1 must be between 0 and "
                f"{max_upper_edges}."
            )

    if omega_ref > lambda_min_sigma:
        return result_tuple(None, None, np.inf)

    nested_edges = None
    if previous_support_mask is not None:
        allowed_edges = (
            upper_triangular_edges(n) if support_scope == "upper"
            else off_diagonal_edges(n)
        )
        previous_support_mask = validate_support_mask(
            previous_support_mask, n, n_edge - 1, allowed_edges,
        )
        nested_edges = [edge for edge in allowed_edges if not previous_support_mask[edge]]

        def nested_supports(n_arg, n_edge_arg):
            for edge in nested_edges:
                mask = previous_support_mask.copy()
                mask[edge] = True
                yield mask

        support_iterator = nested_supports

    best_obj = np.inf
    best_Lambda = None
    best_omega = None
    best_mask = None
    if support_iterator is None:
        if support_scope == "upper":
            support_iterator = get_upper_triangular_supports
        else:
            support_iterator = get_all_supports

    def iter_support_masks():
        if callable(support_iterator):
            yield from support_iterator(n, n_edge)
        else:
            yield from support_iterator

    def iter_support_tasks():
        for support_index, mask in enumerate(iter_support_masks()):
            yield (
                support_index,
                Sigma,
                mask,
                beta,
                max_iter,
                tol,
                zero_tol,
                max_restarts,
                omega_ref,
                None if random_seed is None else random_seed + support_index,
                init_strategy,
            )

    if n_jobs is None:
        max_workers = os.cpu_count() or 1
    else:
        max_workers = n_jobs

    if max_workers is not None and max_workers < 1:
        raise ValueError("n_jobs must be positive or None.")

    support_count = None
    if nested_edges is not None:
        support_count = len(nested_edges)
    elif support_iterator is get_all_supports:
        support_count = comb(n * (n - 1), n_edge)
    elif support_iterator is get_upper_triangular_supports:
        support_count = comb(n * (n - 1) // 2, n_edge)

    if max_workers == 1 or support_count == 1:
        support_results = map(solve_support_worker, iter_support_tasks())
    else:
        support_results = iter_parallel_support_results(
            iter_support_tasks(),
            max_workers,
        )

    best_index = None
    for support_index, mask, result in support_results:
        if previous_support_mask is not None:
            # Break exact ties by enumeration order, including parallel runs.
            if result is not None and (
                best_index is None or (result[2], support_index) < (best_obj, best_index)
            ):
                best_Lambda, best_omega, best_obj = result
                best_Lambda, best_mask = best_Lambda.copy(), mask.copy()
                best_index = support_index
            continue
        best_Lambda, best_omega, best_obj, best_mask = update_best_support(
            best_Lambda,
            best_omega,
            best_obj,
            best_mask,
            mask,
            result,
            obj_tol,
        )

    if best_mask is not None:
        metadata["selected_support_mask"] = best_mask.copy()
        metadata["selected_support_edges"] = support_edges_from_mask(best_mask)

    return result_tuple(best_Lambda, best_omega, best_obj)
