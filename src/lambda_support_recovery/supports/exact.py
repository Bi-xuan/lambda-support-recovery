"""Exact exhaustive support enumeration for small node counts."""

from itertools import combinations
from math import comb

from .common import off_diagonal_edges, support_mask_from_edges, upper_triangular_edges, validate_n_edge


def get_all_supports(n, n_edge):
    """
    Yields (n x n) binary masks where:
    - all diagonal entries are always True (unconstrained)
    - exactly n_edge off-diagonal entries are True (enumerated over all combinations)
    """
    validate_n_edge(n, n_edge)
    off_diag = off_diagonal_edges(n)
    for chosen in combinations(off_diag, n_edge):
        yield support_mask_from_edges(n, chosen)


def get_upper_triangular_supports(n, n_edge):
    """
    Yield masks with exactly n_edge entries strictly above the diagonal.

    Diagonal entries remain True (unconstrained), while every entry below the
    diagonal is False.
    """
    max_edges = n * (n - 1) // 2
    validate_n_edge(n, n_edge, max_edges=max_edges)
    candidates = upper_triangular_edges(n)
    for chosen in combinations(candidates, n_edge):
        yield support_mask_from_edges(n, chosen)
