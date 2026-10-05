# Numerical regression cases adapted from MS-S.
"""Regression checks for screening, fixed-model calibration and CLI wiring."""
from dataclasses import asdict
import json
import numpy as np
import pytest
import lambda_support_recovery.selection as bootstrap
from lambda_support_recovery.selection import DimensionPath, ranked_plateaus

@pytest.fixture
def curve():
    masks = np.repeat(np.eye(3, dtype=bool)[None], 5, axis=0)
    for (i, edge) in enumerate([(0, 1), (0, 2), (1, 2), (1, 0)], 1):
        masks[i:, edge[0], edge[1]] = True
    return dict(dimensions=np.arange(1, 6), objectives=np.array([100.0, 30.0, 5.0, 1.0, 0.0]), penalties=np.arange(1, 6), raw_objectives=np.array([100.0, 30.0, 5.0, 1.0, 0.0]), Sigma=np.eye(3) * 2, support_masks=masks, support_valid=np.ones(5, bool), num_samples=100)

@pytest.fixture
def mock_fits(monkeypatch, curve):
    calls = []

    def fit(sigma, mask, settings, **kwargs):
        dimension = int(mask.sum()) - 2
        calls.append(dimension)
        return (np.eye(3) * 0.5, 1.0, curve['raw_objectives'][dimension - 1])
    monkeypatch.setattr(bootstrap, '_fit', fit)
    return calls

def test_bounded_plateau_ranking_excludes_endpoints_and_breaks_ties():
    path = DimensionPath((0.0, 1.0, 4.0, 16.0), (5, 4, 3, 1))
    assert [p.dimension for p in ranked_plateaus(path)] == [3, 4]

@pytest.mark.parametrize('alpha, expected', [(0.05, 2), (0.051, 3)])
def test_upper_tail_plus_one_strict_threshold_and_sequential_nulls(monkeypatch, curve, mock_fits, alpha, expected):
    gains = iter([4.0] * 19 + [0.0] * 19)
    monkeypatch.setattr(bootstrap, '_bootstrap_gain', lambda task: next(gains))
    result = bootstrap.select_plateau_bootstrap(**curve, bootstrap_replicates=19, alpha=alpha)
    assert result.candidate_dimensions == (4, 3, 2)
    assert result.selected_dimension == expected
    assert [c.p_value for c in result.comparisons] == [1.0, 0.05]
    assert [c.smaller_dimension for c in result.comparisons] == [3, 2]
    assert mock_fits == [3, 4, 2]

def test_significant_first_pair_stops_and_truth_only_affects_precision(monkeypatch, curve, mock_fits):
    monkeypatch.setattr(bootstrap, '_bootstrap_gain', lambda task: 0.0)
    result = bootstrap.select_plateau_bootstrap(**curve, bootstrap_replicates=99, true_support=curve['support_masks'][2])
    assert result.selected_dimension == 4
    assert result.precision == pytest.approx(2 / 3)
    assert len(result.comparisons) == 1
    assert mock_fits == [3, 4]

def test_top_k_one_and_more_than_available(monkeypatch, curve, mock_fits):
    result = bootstrap.select_plateau_bootstrap(**curve, top_plateaus=1)
    assert result.candidate_dimensions == (3,)
    assert result.selected_dimension == 3
    assert not result.comparisons and (not mock_fits)
    monkeypatch.setattr(bootstrap, '_bootstrap_gain', lambda task: 1000000.0)
    result = bootstrap.select_plateau_bootstrap(**curve, top_plateaus=20, bootstrap_replicates=1)
    assert result.candidate_dimensions == (4, 3, 2)
    assert result.selected_dimension == 2

@pytest.mark.parametrize('option,value', [('top_plateaus', 0), ('top_plateaus', 1.5), ('bootstrap_replicates', 0), ('alpha', 0), ('alpha', 1), ('alpha', np.nan), ('n_jobs', 0), ('num_samples', -1), ('seed', -1)])
def test_invalid_configuration(curve, option, value):
    with pytest.raises(ValueError):
        bootstrap.select_plateau_bootstrap(**{**curve, option: value})

def test_reject_non_nested_candidates_before_bootstrapping(curve):
    curve['support_masks'][1, 0, 1] = False
    curve['support_masks'][1, 2, 0] = True
    with pytest.raises(ValueError, match='must be nested'):
        bootstrap.select_plateau_bootstrap(**curve)

def test_refit_mismatch_rejects_calibration(monkeypatch, curve):
    monkeypatch.setattr(bootstrap, '_fit', lambda *a, **k: (np.eye(3) / 2, 1.0, 999.0))
    with pytest.raises(ValueError, match='differs from the saved curve'):
        bootstrap.select_plateau_bootstrap(**curve)

def test_no_bounded_plateau_fails(curve):
    curve['objectives'] = np.arange(5.0, 0.0, -1)
    with pytest.raises(ValueError, match='bounded plateau'):
        bootstrap.select_plateau_bootstrap(**curve)

def test_both_fixed_masks_refit_same_sample_and_negative_gain_clipped(monkeypatch, curve):
    calls = []

    def solver(sigma, mask, **kwargs):
        calls.append((sigma, mask, kwargs))
        return (np.eye(3) / 2, 1.0, float(len(calls)))
    monkeypatch.setattr(bootstrap, 'solve_support_with_restarts', solver)
    (small, large) = curve['support_masks'][1:3]
    sigma = np.eye(3) * 0.9
    gain = bootstrap._bootstrap_gain((sigma, small, large, bootstrap.FitSettings()))
    assert gain == 0
    assert len(calls) == 2
    assert calls[0][0] is sigma and calls[1][0] is sigma
    assert calls[0][1] is small and calls[1][1] is large
    assert all((c[2]['omega_upper'] is None and c[2]['omega_fixed'] == 1 for c in calls))

def test_stationary_covariance_orientation_and_unstable_null():
    lam = np.array([[0.2, 0.4], [0.0, -0.3]])
    (sigma, _) = bootstrap._null_covariance((lam, 0.7, 0.0))
    np.testing.assert_allclose(sigma - lam.T @ sigma @ lam, np.eye(2) * 0.7, atol=1e-14)
    with pytest.raises(ValueError, match='spectral radius'):
        bootstrap._null_covariance((np.eye(2), 1.0, 0.0))

def test_bootstrap_covariances_use_original_N_and_deterministic_seed(monkeypatch, curve, mock_fits):
    observed = []
    monkeypatch.setattr(bootstrap, '_bootstrap_gain', lambda task: observed.append(task[0].copy()) or 1000000.0)
    bootstrap.select_plateau_bootstrap(**curve, bootstrap_replicates=3)
    first = np.array(observed)
    observed.clear()
    bootstrap.select_plateau_bootstrap(**curve, bootstrap_replicates=3)
    np.testing.assert_array_equal(first, observed)
    assert not np.array_equal(first[0], first[1])

def test_native_refits_serial_and_parallel_agree(curve):
    settings = bootstrap.FitSettings(max_iter=8, max_restarts=1)
    curve['raw_objectives'] = np.array([bootstrap._fit(curve['Sigma'], mask, settings)[2] for mask in curve['support_masks']])
    serial = bootstrap.select_plateau_bootstrap(**curve, fit_settings=settings, bootstrap_replicates=2)
    parallel = bootstrap.select_plateau_bootstrap(**curve, fit_settings=settings, bootstrap_replicates=2, n_jobs=2)
    assert serial == parallel
