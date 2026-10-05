# Numerical regression cases adapted from MS-S.
"""Forward-nested enumeration, opt-out, failures, and checkpoint safety."""
import json
import numpy as np
import pytest
from lambda_support_recovery import curve as curve
from lambda_support_recovery.optimizers import support_search as search
from lambda_support_recovery.supports.common import off_diagonal_edges, support_mask_from_edges
(A, B, C) = ((0, 1), (0, 2), (1, 2))

@pytest.fixture
def scored_supports(monkeypatch):
    visited = []
    scores = {frozenset(): 10.0, frozenset([A]): 1.0, frozenset([B]): 2.0, frozenset([C]): 3.0, frozenset([A, B]): 0.6, frozenset([A, C]): 0.3, frozenset([B, C]): 0.1, frozenset([A, B, C]): 0.01}

    def worker(task):
        (index, sigma, mask) = task[:3]
        edges = frozenset(search.support_edges_from_mask(mask))
        visited.append(edges)
        return (index, mask, (np.zeros_like(sigma), 0.1, scores[edges]))
    monkeypatch.setattr(search, 'solve_support_worker', worker)
    return visited

def test_default_nested_path_differs_from_unrestricted_minima(scored_supports):
    nested = curve.compute_objective_curve(np.eye(3), support_scope='upper')
    assert len(scored_supports) == 1 + 3 + 2 + 1
    assert search.support_edges_from_mask(nested[4][2]) == [A, C]
    assert nested[1][2] == 0.3
    scored_supports.clear()
    independent = curve.compute_objective_curve(np.eye(3), support_scope='upper', nested_supports=False)
    assert len(scored_supports) == 1 + 3 + 3 + 1
    assert search.support_edges_from_mask(independent[4][2]) == [B, C]
    assert independent[1][2] == 0.1

@pytest.mark.parametrize('scope,preselected', [('upper', None), ('all', None), ('all', [(2, 0), A, C])])
def test_extensions_obey_scope_and_preselection(monkeypatch, scope, preselected):
    seen = []

    def worker(task):
        (index, sigma, mask) = task[:3]
        seen.append(mask.copy())
        return (index, mask, (np.zeros_like(sigma), 0.1, float(index)))
    monkeypatch.setattr(search, 'solve_support_worker', worker)
    previous = support_mask_from_edges(3, [A])
    result = search.optimize_lambda(np.eye(3), 3, previous_support_mask=previous, support_scope=scope, preselect_edges=preselected, return_metadata=True)
    allowed = preselected if preselected is not None else [A, B, C] if scope == 'upper' else off_diagonal_edges(3)
    assert len(seen) == len(allowed) - 1
    for mask in seen:
        assert np.all(~previous | mask)
        assert len(search.support_edges_from_mask(mask)) == 2
        assert set(search.support_edges_from_mask(mask)) <= set(allowed)
    assert np.array_equal(result[3]['selected_support_mask'], seen[0])

@pytest.mark.parametrize('difference', [0.0, 1e-10])
def test_nested_strict_minimum_and_parallel_tie_order(monkeypatch, difference):

    def worker(task):
        (index, sigma, mask) = task[:3]
        return (index, mask, (np.zeros_like(sigma), 0.1, 1.0 - index * difference))
    monkeypatch.setattr(search, 'solve_support_worker', worker)
    monkeypatch.setattr(search, 'iter_parallel_support_results', lambda tasks, workers: reversed([worker(t) for t in tasks]))
    result = search.optimize_lambda(np.eye(3), 3, support_scope='upper', n_jobs=2, previous_support_mask=support_mask_from_edges(3, [A]), return_metadata=True)
    assert result[3]['selected_support_edges'] == ([A, B] if difference == 0 else [A, C])

def test_resume_uses_the_saved_predecessor(scored_supports):
    full = curve.compute_objective_curve(np.eye(3), support_scope='upper')
    prefix = tuple((a[:2] if i in (0, 1, 4, 5) else a for (i, a) in enumerate(full)))
    scored_supports.clear()
    resumed = curve.compute_objective_curve(np.eye(3), support_scope='upper', initial_curve_result=prefix)
    assert len(scored_supports) == 3
    for (actual, expected) in zip(resumed, full):
        np.testing.assert_array_equal(actual, expected)

@pytest.mark.parametrize('problem', ['gap', 'not_nested', 'missing_masks', 'outside_scope', 'failed_predecessor'])
def test_reject_invalid_nested_prefix(scored_supports, problem):
    full = curve.compute_objective_curve(np.eye(3), support_scope='upper')
    prefix = [a[:3].copy() if i in (0, 1, 4, 5) else a.copy() for (i, a) in enumerate(full)]
    if problem == 'gap':
        prefix[0][1] = 3
    elif problem == 'not_nested':
        prefix[4][2] = support_mask_from_edges(3, [B, C])
    elif problem == 'missing_masks':
        prefix = prefix[:4]
    elif problem == 'outside_scope':
        prefix[4][1] = support_mask_from_edges(3, [(2, 0)])
    else:
        prefix[5][1] = False
        prefix[1][1] = np.inf
    with pytest.raises(ValueError):
        curve.compute_objective_curve(np.eye(3), support_scope='upper', initial_curve_result=tuple(prefix))

def test_failed_dimension_blocks_later_extensions_and_can_be_resumed(monkeypatch):
    calls = []

    def optimize(sigma, dim, **kwargs):
        calls.append(dim)
        if dim == 2:
            return (None, None, np.inf, {})
        assert dim == 1
        return (np.zeros_like(sigma), 0.1, 1.0, {'selected_support_mask': np.eye(3, dtype=bool)})
    monkeypatch.setattr(curve, 'optimize_lambda', optimize)
    saved = []
    result = curve.compute_objective_curve(np.eye(3), support_scope='upper', save_callback=saved.append)
    assert calls == [1, 2]
    assert len(saved) == 4
    assert result[5].tolist() == [True, False, False, False]
    assert np.all(np.isposinf(result[1][1:]))
    resumed = curve.compute_objective_curve(np.eye(3), support_scope='upper', initial_curve_result=result)
    assert calls == [1, 2]
    np.testing.assert_array_equal(resumed[5], result[5])
