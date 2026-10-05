# Numerical regression cases adapted from MS-S.
"""Regression tests for theorem-based penalty constants."""
from math import log, log1p, sqrt
from pathlib import Path
import sys
import pytest
from scipy.integrate import quad
from lambda_support_recovery.penalty import PenaltyConstants, compute_K, compute_L_t_xi, compute_R, compute_c, compute_t, compute_v, pen_n, theorem_constants

def entropy_integrand(u):
    if u == 0.0:
        return float('inf')
    return sqrt(log1p(1.0 / u))

def test_updated_theorem_penalty_formulas_and_default_r():
    constants = PenaltyConstants(num_samples=100, n=5, lambda_sum=4.0, lambda_inf_norm=2.0, lambda_2_norm=3.0, sigma_fro_norm=6.0, sigma_op_norm=4.0, sigma_trace=10.0, Lm=1.5, L=0.75, xi=2.0)
    log_term = log(constants.num_samples) + constants.xi
    expected_R = sqrt(constants.lambda_sum + 2.0 * constants.lambda_inf_norm * log_term) + 2.0 * constants.lambda_2_norm * sqrt(log_term)
    expected_c = 16.0 / constants.num_samples * (1.0 + 1.0 / constants.num_samples) * constants.L ** 2 * (1.0 + constants.L ** 2) * expected_R ** 4 + 8.0 / constants.num_samples * sqrt(constants.n) * (1.0 + 2.0 * constants.L ** 2) * constants.r * expected_R ** 2
    expected_v = constants.num_samples * expected_c ** 2 / 2.0
    first_t_term = constants.xi * (constants.sigma_op_norm + expected_R ** 2) / (3.0 * constants.num_samples)
    expected_t = first_t_term + sqrt(first_t_term ** 2 + 2.0 * constants.xi * constants.sigma_op_norm * expected_R ** 2 / constants.num_samples)
    expected_L_t_xi = 4.0 * expected_t * constants.L * ((3.0 * constants.sigma_op_norm + expected_R ** 2) * (1.0 + constants.L ** 2) + constants.r) + 2.0 * constants.n * expected_t * (1.0 + constants.L ** 2) + 4.0 / constants.num_samples * (2.0 * constants.sigma_op_norm ** 2 * constants.L ** 3 + constants.sigma_fro_norm ** 2 * constants.L + constants.sigma_op_norm ** 2 * constants.L)
    (integral, _) = quad(entropy_integrand, 0.0, 1.0 / (8.0 * expected_L_t_xi), points=[0.0])
    expected_K = 96.0 * expected_L_t_xi ** 2 * (constants.L + constants.r) * integral
    expected_penalty = sqrt(3.0) * (expected_K + sqrt(2.0 * expected_v * constants.Lm))
    assert constants.r == 1.0
    assert compute_R(constants) == pytest.approx(expected_R)
    assert compute_c(constants) == pytest.approx(expected_c)
    assert compute_v(constants) == pytest.approx(expected_v)
    assert compute_t(constants) == pytest.approx(expected_t)
    assert compute_L_t_xi(constants) == pytest.approx(expected_L_t_xi)
    assert compute_K(constants) == pytest.approx(expected_K)
    assert pen_n(3.0, constants) == pytest.approx(expected_penalty)

def test_theorem_constants_report_updated_names():
    constants = PenaltyConstants(num_samples=20, n=4, lambda_sum=2.0, lambda_inf_norm=1.0, lambda_2_norm=1.5, sigma_fro_norm=3.0, sigma_op_norm=2.0, sigma_trace=4.0)
    derived = theorem_constants(constants)
    assert 't' in derived
    assert 'L_t_xi' in derived
    assert 'L_prime' not in derived
    assert 'B' not in derived
