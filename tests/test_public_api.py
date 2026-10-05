"""Public API integration, configurable penalty references, and validation."""
from dataclasses import replace

import numpy as np
import pytest

from lambda_support_recovery import (
    PenaltyConfig, compute_support_curve, select_support, select_from_curve,
    build_penalty_constants,
)
from lambda_support_recovery import api
from lambda_support_recovery.penalty import pen_n
from lambda_support_recovery.selection import _fit
from lambda_support_recovery import FitSettings


@pytest.fixture(scope="module")
def sigma():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 4))
    return a @ a.T / 4 + np.eye(4)


@pytest.fixture(scope="module")
def curve(sigma):
    return compute_support_curve(sigma, num_samples=1000, max_iter=120, max_restarts=2, support_scope="upper")


def test_default_function_returns_mask_and_diagnostics_are_optional(sigma, curve, capsys):
    result = select_support(sigma, num_samples=1000, max_iter=120, max_restarts=2,
                            support_scope="upper", bootstrap_replicates=19, return_result=True)
    mask = select_support(sigma, num_samples=1000, max_iter=120, max_restarts=2,
                          support_scope="upper", bootstrap_replicates=19)
    np.testing.assert_array_equal(mask, result.support)
    assert mask.dtype == bool and mask.shape == sigma.shape
    assert np.all(np.diag(mask))
    assert result.selected_dimension == 1 + len(result.selected_edges)
    assert result.lm_mode == "constant" and np.all(result.lm_values == 1)
    assert len(result.bootstrap_comparisons) == 2
    assert result.curve.fit_settings.max_restarts == 2
    assert result.diagnostics.fit_settings == result.curve.fit_settings
    assert result.resolved_omega_ref == pytest.approx(0.93 * np.linalg.eigvalsh(sigma)[0])
    assert capsys.readouterr().out == ""


def test_cached_plain_plateau_support_counts_and_weight(curve, monkeypatch):
    monkeypatch.setattr(api, "compute_support_curve", lambda *a, **k: pytest.fail("Curve should not be rebuilt"))
    result = select_from_curve(curve, method="Plateau", lm_weight=0.2, return_result=True)
    np.testing.assert_allclose(result.lm_values, np.array([1, 6, 15, 20, 15, 6, 1]) * 0.2)
    assert not result.bootstrap_comparisons
    assert result.diagnostics.recommendation_factor == 2
    assert result.lm_mode == "support-count"


def test_known_omega_and_population_metadata_do_not_change_default_fit(sigma):
    first = compute_support_curve(sigma, max_restarts=1, max_iter=20, support_scope="upper", omega_star=0.1)
    second = compute_support_curve(sigma, max_restarts=1, max_iter=20, support_scope="upper", omega_star=0.8)
    np.testing.assert_array_equal(first.support_masks, second.support_masks)
    np.testing.assert_array_equal(first.raw_objectives, second.raw_objectives)
    known = compute_support_curve(sigma, max_restarts=1, max_iter=20, support_scope="upper",
                                  omega_ref=0.4, omega_star=0.4, fit_omega_ref=False)
    np.testing.assert_allclose(known.fitted_omegas[known.support_valid], 0.4)
    assert known.fit_settings.omega_fixed == 0.4


def test_fixed_reference_can_equal_observed_minimum_eigenvalue():
    fitted = _fit(np.eye(2), np.eye(2, dtype=bool), FitSettings(omega_fixed=1, max_iter=20, max_restarts=1))
    assert fitted[1] == 1
    assert np.isfinite(fitted[2])


def test_penalty_defaults_use_observed_covariance(sigma):
    constants = build_penalty_constants(sigma, 1000)
    eigenvalues = np.linalg.eigvalsh(sigma)
    assert constants.lambda_sum == pytest.approx(eigenvalues.sum())
    assert constants.lambda_inf_norm == pytest.approx(eigenvalues[-1])
    assert constants.lambda_2_norm == pytest.approx(np.linalg.norm(eigenvalues))
    assert constants.sigma_fro_norm == pytest.approx(np.linalg.norm(sigma, "fro"))
    assert constants.sigma_op_norm == pytest.approx(np.linalg.norm(sigma, 2))
    assert constants.sigma_trace == pytest.approx(np.trace(sigma))
    assert constants.L == 1 and constants.r == 1 and constants.xi == 10


def test_penalty_matrix_scalar_and_individual_overrides(sigma):
    scalar = build_penalty_constants(sigma, 1000, config=PenaltyConfig(sigma=2.0, lambda_bound=0.75))
    matrix = build_penalty_constants(sigma, 1000, config=PenaltyConfig(sigma=np.eye(4) * 2, lambda_bound=0.75))
    assert scalar == matrix
    assert scalar.lambda_sum == 8 and scalar.lambda_inf_norm == 2
    overridden = build_penalty_constants(sigma, 1000, config=PenaltyConfig(
        lambda_bound=0.5, noise_bound=2, xi=3,
        lambda_sum=4, lambda_inf_norm=2, lambda_2_norm=3,
        sigma_fro_norm=6, sigma_op_norm=4, sigma_trace=10,
    ))
    assert (overridden.L, overridden.r, overridden.xi) == (0.5, 2, 3)
    assert (overridden.lambda_sum, overridden.lambda_inf_norm, overridden.lambda_2_norm) == (4, 2, 3)
    assert (overridden.sigma_fro_norm, overridden.sigma_op_norm, overridden.sigma_trace) == (6, 4, 10)


def test_penalty_overrides_change_penalty_but_not_cached_fits(curve):
    original = curve.raw_objectives.copy()
    first = select_from_curve(curve, method="plateau", return_result=True)
    second = select_from_curve(curve, method="plateau", penalty_config=PenaltyConfig(sigma=1.0, lambda_bound=0.7), return_result=True)
    assert not np.allclose(first.penalty_values, second.penalty_values)
    np.testing.assert_array_equal(original, curve.raw_objectives)
    assert second.curve is curve
    for d, lm, value in zip(curve.dimensions, second.lm_values, second.penalty_values):
        assert value == pytest.approx(pen_n(float(d), second.penalty_constants, Lm=float(lm)))


def test_model_support_is_not_thresholded_coefficients(curve):
    zero_fits = replace(curve, fitted_lambdas=np.zeros_like(curve.fitted_lambdas))
    result = select_from_curve(zero_fits, method="plateau", return_result=True)
    assert result.selected_edges
    assert np.count_nonzero(result.fitted_lambda) == 0
    assert np.count_nonzero(result.support) == len(curve.sigma_hat) + result.selected_dimension - 1


def test_floor_only_changes_screening_and_preserves_raw_curve(curve):
    raw = curve.raw_objectives.copy()
    result = select_from_curve(curve, method="plateau", objective_floor=0.03, return_result=True)
    assert np.all(result.screened_objectives[np.isfinite(raw) & (raw <= 0.03)] == 0)
    np.testing.assert_array_equal(curve.raw_objectives, raw)


def test_numpy_random_state_is_preserved(sigma):
    np.random.seed(73)
    state = np.random.get_state()
    compute_support_curve(sigma, max_iter=8, max_restarts=1, support_scope="upper", init_strategy="random")
    actual = np.random.random(4)
    np.random.set_state(state)
    np.testing.assert_array_equal(actual, np.random.random(4))


@pytest.mark.parametrize("options", [
    {"num_samples": 1}, {"num_samples": True}, {"max_restarts": 0},
    {"omega_ref": 0.4}, {"omega_ref": 100, "fit_omega_ref": False},
    {"method": "unknown"}, {"lm_weight": 0}, {"lm_mode": "support-count"},
    {"nested_supports": False}, {"init_strategy": "random"},
    {"bootstrap_alpha": 1}, {"bootstrap_replicates": 0}, {"top_plateaus": 0},
    {"bootstrap_seed": -1}, {"n_jobs": 0}, {"objective_floor": np.nan},
    {"max_iter": 0}, {"tol": 0}, {"kappa": 1.1}, {"fit_omega_ref": "true"},
    {"penalty_config": PenaltyConfig(lambda_bound=3)},
])
def test_invalid_options_fail_before_numerical_search(sigma, options):
    kwargs = dict(num_samples=1000, max_restarts=1, max_iter=5)
    kwargs.update(options)
    with pytest.raises((ValueError, TypeError)):
        select_support(sigma, **kwargs)


@pytest.mark.parametrize("matrix", [np.eye(1), np.ones((2, 3)), np.diag([1., 0.]), np.diag([1., -1.]), np.array([[1., 1.], [0., 1.]]), np.eye(2) * np.nan, np.eye(2, dtype=complex)])
def test_invalid_covariances(matrix):
    with pytest.raises(ValueError):
        select_support(matrix, num_samples=1000)


@pytest.mark.parametrize("options", [
    {"lambda_bound": 0}, {"lambda_bound": np.inf}, {"noise_bound": -1},
    {"xi": 0}, {"lambda_sum": -1}, {"sigma_op_norm": np.nan},
])
def test_invalid_penalty_bounds(options):
    with pytest.raises(ValueError):
        PenaltyConfig(**options)


def test_sample_size_cannot_be_changed_on_cached_curve(curve):
    with pytest.raises(ValueError, match="original sample size"):
        select_from_curve(curve, num_samples=2000)


def test_no_bounded_plateau_is_an_explicit_failure():
    with pytest.raises(ValueError, match="bounded plateau"):
        select_support(np.eye(2), num_samples=100, max_iter=20, max_restarts=1)
