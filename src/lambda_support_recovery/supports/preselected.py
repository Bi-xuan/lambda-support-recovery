"""Support enumeration restricted to a preselected off-diagonal edge set."""

from itertools import combinations
from math import comb

from .common import support_mask_from_edges, validate_n_edge


def normalize_preselected_edges(n, selected_edges):
    """Validate and normalize directed off-diagonal edge tuples."""
    normalized = []
    seen = set()

    for edge in selected_edges:
        if len(edge) != 2:
            raise ValueError("Each preselected edge must have two indices.")

        i, j = int(edge[0]), int(edge[1])
        if i == j:
            raise ValueError("Preselected edges must be off-diagonal.")
        if not (0 <= i < n and 0 <= j < n):
            raise ValueError("Preselected edge index out of range.")
        if (i, j) in seen:
            raise ValueError("Preselected edges must be unique.")

        seen.add((i, j))
        normalized.append((i, j))

    return normalized


def count_preselected_supports(n_edge, selected_edges):
    if n_edge > len(selected_edges):
        return 0
    return comb(len(selected_edges), n_edge)


def get_preselected_supports(n, n_edge, selected_edges):
    """
    Yield support masks with diagonal entries and n_edge chosen preselected edges.

    Edges are directed, matching supports.common.off_diagonal_edges.
    """
    validate_n_edge(n, n_edge)
    selected_edges = normalize_preselected_edges(n, selected_edges)

    if n_edge > len(selected_edges):
        raise ValueError(
            "n_edge cannot exceed the number of preselected edges: "
            f"n_edge={n_edge}, preselected={len(selected_edges)}."
        )

    for chosen in combinations(selected_edges, n_edge):
        yield support_mask_from_edges(n, chosen)
