"""Fixed-omega ADMM solver and covariance-model simulation helpers."""

import numpy as np

from .config import nonnegative_float

DEFAULT_OMEGA_STAR = 0.1


def van_der_corput(index, base):
    value = 0.0
    factor = 1.0 / base

    while index > 0:
        value += factor * (index % base)
        index //= base
        factor /= base

    return value


def first_primes(count):
    primes = []
    candidate = 2

    while len(primes) < count:
        is_prime = True
        for prime in primes:
            if prime * prime > candidate:
                break
            if candidate % prime == 0:
                is_prime = False
                break

        if is_prime:
            primes.append(candidate)

        candidate += 1

    return primes


def halton_point(index, dim):
    primes = first_primes(dim)
    return np.array([
        van_der_corput(index, base)
        for base in primes
    ])


def impose_support(M, mask):
    M_out = M.copy()
    M_out[~mask] = 0.0
    return M_out


def covariance_from_lambda_star(Lambda_star, omega):
    if omega < 0.0:
        raise ValueError("omega must be nonnegative.")
    if lambda_star_spectral_radius(Lambda_star) >= 1.0:
        raise ValueError(
            "All eigenvalues of Lambda_star must be smaller than 1 in absolute value."
        )

    n = Lambda_star.shape[0]
    system_matrix = np.eye(n * n) - np.kron(Lambda_star.T, Lambda_star.T)
    rhs = (omega * np.eye(n)).reshape(-1, order="F")
    sigma_vec = np.linalg.solve(system_matrix, rhs)
    Sigma = sigma_vec.reshape((n, n), order="F")
    return 0.5 * (Sigma + Sigma.T)


def lambda_star_spectral_radius(Lambda_star):
    return np.max(np.abs(np.linalg.eigvals(Lambda_star)))


def sample_lambda_star_from_mask(mask, target_spectral_radius=0.5, seed=0):
    if target_spectral_radius >= 1.0:
        raise ValueError("target_spectral_radius must be smaller than 1.")
    if target_spectral_radius <= 0.0:
        raise ValueError("target_spectral_radius must be positive.")
    if not np.any(mask):
        raise ValueError("Lambda_star_mask must contain at least one True entry.")

    rng = np.random.default_rng(seed)
    Lambda_star = np.zeros(mask.shape)
    Lambda_star[mask] = rng.normal(size=np.count_nonzero(mask))
    spectral_radius = lambda_star_spectral_radius(Lambda_star)
    if spectral_radius == 0.0:
        raise ValueError("Cannot scale Lambda_star with zero spectral radius.")

    Lambda_star *= target_spectral_radius / spectral_radius
    return Lambda_star


def is_finite_state(*arrays):
    return all(np.all(np.isfinite(arr)) for arr in arrays)


def initialize_admm_state(n, support_mask, restart_index, init_strategy):
    if init_strategy == "halton":
        point = halton_point(restart_index + 1, n * n)
        init = 2.0 * (point.reshape(n, n) - 0.5)
        L1 = impose_support(init, support_mask)
        L2 = impose_support(init.copy(), support_mask)
        return L1, L2

    if init_strategy == "random":
        L1 = impose_support(np.random.randn(n, n), support_mask)
        L2 = impose_support(np.random.randn(n, n), support_mask)
        return L1, L2

    raise ValueError(
        "init_strategy must be one of {'halton', 'random'}."
    )


def admm_solve(
    Sigma,
    support_mask,
    beta=1.0,
    max_iter=500,
    tol=1e-6,
    max_restarts=3,
    *,
    omega_ref,
    init_strategy="halton",
    init_offset=0,
):
    """Fit Lambda on a support while holding omega_ref constant."""
    omega_ref = nonnegative_float(omega_ref, "omega_ref")

    n = Sigma.shape[0]
    identity = np.eye(n)
    target = Sigma - omega_ref * identity
    best_Lambda = None
    best_obj = np.inf

    for restart_index in range(max_restarts):
        # Initialize
        L1, L2 = initialize_admm_state(
            n,
            support_mask,
            init_offset + restart_index,
            init_strategy,
        )
        alpha = np.zeros((n, n))
        failed = False

        for _ in range(max_iter):
            L1_prev = L1.copy()

            try:
                # Update Lambda_1
                SL2 = Sigma @ L2
                A1 = 2 * SL2 @ SL2.T + beta * identity
                B1 = 2 * SL2 @ target - alpha + beta * L2
                L1 = np.linalg.solve(A1, B1)
                L1 = impose_support(L1, support_mask)

                # Update Lambda_2
                SL1 = Sigma @ L1
                A2 = 2 * SL1 @ SL1.T + beta * identity
                B2 = 2 * SL1 @ target + alpha + beta * L1
                L2 = np.linalg.solve(A2, B2)
                L2 = impose_support(L2, support_mask)
            except np.linalg.LinAlgError:
                failed = True
                break

            if not is_finite_state(L1, L2, alpha):
                failed = True
                break

            # Update dual variable
            alpha = alpha + beta * (L1 - L2)

            if not is_finite_state(alpha):
                failed = True
                break

            # Check convergence
            if np.linalg.norm(L1 - L1_prev, 'fro') < tol:
                break

        if not failed and is_finite_state(L1, L2, alpha):
            residual = target - L1.T @ Sigma @ L1
            obj = np.linalg.norm(residual, 'fro') ** 2
            if np.isfinite(obj) and obj < best_obj:
                best_Lambda = L1.copy()
                best_obj = obj

    if best_Lambda is not None:
        return best_Lambda, omega_ref

    nan_matrix = np.full((n, n), np.nan)
    return nan_matrix, np.nan
