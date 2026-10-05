# Numerical regression cases adapted from MS-S.
"""
Checks for preselected support enumeration and optimizer integration.

Run from the project root:
    python -m pytest tests/test_preselected_supports.py
    python tests/test_preselected_supports.py
"""
from math import comb
from pathlib import Path
import sys
import numpy as np
import pytest
from lambda_support_recovery.optimizers.support_search import optimize_lambda, select_preselected_edge_scores
from lambda_support_recovery.curve import compute_objective_curve
from lambda_support_recovery.supports.preselected import get_preselected_supports

def off_diagonal_edges_in_mask(mask):
    n = mask.shape[0]
    return {(i, j) for i in range(n) for j in range(n) if i != j and mask[i, j]}

def test_preselected_supports_only_use_selected_edges():
    n = 4
    n_edge = 2
    selected_edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    supports = list(get_preselected_supports(n, n_edge, selected_edges))
    assert len(supports) == comb(len(selected_edges), n_edge)
    for mask in supports:
        assert np.all(np.diag(mask))
        assert off_diagonal_edges_in_mask(mask) <= set(selected_edges)
        assert len(off_diagonal_edges_in_mask(mask)) == n_edge

def test_preselected_supports_reject_too_few_edges():
    with pytest.raises(ValueError, match='n_edge cannot exceed'):
        list(get_preselected_supports(3, 2, [(0, 1)]))

def test_optimize_lambda_keeps_three_value_return_by_default():
    Sigma = np.eye(3)
    result = optimize_lambda(Sigma, D_m=1, max_iter=2, max_restarts=1, n_jobs=1, random_seed=7, refine_after_fixed_omega=False)
    assert len(result) == 3

def test_optimize_lambda_returns_preselection_metadata_when_requested():
    Sigma = np.eye(3)
    d_m = 2
    (Lambda, omega, obj, metadata) = optimize_lambda(Sigma, D_m=d_m, max_iter=2, max_restarts=1, n_jobs=1, random_seed=7, refine_after_fixed_omega=False, preselect_k=2, return_metadata=True)
    assert Lambda is not None
    assert np.isfinite(omega)
    assert np.isfinite(obj)
    assert metadata['preselect_k'] == 2
    assert len(metadata['preselected_edges']) == 2
    assert len(metadata['preselected_scores']) == 2
    selected_mask = metadata['selected_support_mask']
    selected_edges = metadata['selected_support_edges']
    assert selected_mask.shape == Sigma.shape
    assert np.all(np.diag(selected_mask))
    assert len(off_diagonal_edges_in_mask(selected_mask)) == d_m - 1
    assert set(selected_edges) == off_diagonal_edges_in_mask(selected_mask)

def test_compute_objective_curve_returns_selected_supports_per_d_m():
    Sigma = np.eye(2)
    (d_m_values, objective_values, fallback_d_m_values, fallback_objective_values, selected_support_masks, selected_support_valid) = compute_objective_curve(Sigma, max_iter=2, max_restarts=1, n_jobs=1, random_seed=7, refine_after_fixed_omega=False)
    assert d_m_values.tolist() == [1, 2, 3]
    assert len(objective_values) == len(d_m_values)
    assert len(fallback_d_m_values) == 0
    assert len(fallback_objective_values) == 0
    assert selected_support_masks.shape == (len(d_m_values), 2, 2)
    assert selected_support_valid.shape == (len(d_m_values),)
    assert np.all(selected_support_valid)
    for (d_m, mask) in zip(d_m_values, selected_support_masks):
        assert np.all(np.diag(mask))
        assert len(off_diagonal_edges_in_mask(mask)) == d_m - 1

def test_both_per_pair_keeps_both_directions_for_best_pairs():
    edge_scores = [((3, 2), 0.00754384), ((2, 3), 0.00775762), ((3, 0), 0.07273263), ((0, 3), 0.07593184), ((1, 2), 0.2), ((2, 1), 0.19)]
    selected = select_preselected_edge_scores(edge_scores, preselect_k=2, direction_policy='both_per_pair')
    assert selected == [((3, 2), 0.00754384), ((2, 3), 0.00775762), ((3, 0), 0.07273263), ((0, 3), 0.07593184)]
