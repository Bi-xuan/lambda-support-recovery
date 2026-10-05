"""Shared support-mask helpers used by exact and future preselected searches."""

import numpy as np


def off_diagonal_edges(n):
    return [(i, j) for i in range(n) for j in range(n) if i != j]


def upper_triangular_edges(n):
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def diagonal_mask(n):
    return np.eye(n, dtype=bool)


def support_mask_from_edges(n, edges):
    mask = diagonal_mask(n)
    for i, j in edges:
        if i == j:
            raise ValueError("Support edges must be off-diagonal.")
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("Support edge index out of range.")
        mask[i, j] = True
    return mask


def validate_support_mask(mask, n, n_edge, allowed_edges):
    """Return a mask with the expected dimension and allowed edge positions."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (n, n):
        raise ValueError(f"Support mask must have shape ({n}, {n}).")
    if not np.all(np.diag(mask)) or np.count_nonzero(mask) != n + n_edge:
        raise ValueError(
            f"Support mask must contain all diagonal entries and {n_edge} off-diagonal edges."
        )
    allowed = support_mask_from_edges(n, allowed_edges)
    if np.any(mask & ~allowed):
        raise ValueError("Support mask contains edges outside the allowed support scope.")
    return mask.copy()


def validate_n_edge(n, n_edge, max_edges=None):
    if n < 1:
        raise ValueError("n must be positive.")
    if max_edges is None:
        max_edges = n * (n - 1)
    if not (0 <= n_edge <= max_edges):
        raise ValueError(f"n_edge must be between 0 and {max_edges}.")
