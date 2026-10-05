# Numerical regression cases adapted from MS-S.
"""Check ADMM initialization strategies."""
import argparse
from pathlib import Path
import sys
import numpy as np
from lambda_support_recovery.admm import admm_solve, halton_point, initialize_admm_state

def make_problem():
    Sigma = np.array([[1.4, 0.2, 0.1], [0.2, 1.2, 0.0], [0.1, 0.0, 1.1]])
    support_mask = np.eye(3, dtype=bool)
    return (Sigma, support_mask)

def test_halton_initialization_is_seed_independent():
    (_, support_mask) = make_problem()
    np.random.seed(1)
    (L1_a, L2_a) = initialize_admm_state(3, support_mask, 0, 'halton')
    np.random.seed(999)
    (L1_b, L2_b) = initialize_admm_state(3, support_mask, 0, 'halton')
    assert np.array_equal(L1_a, L1_b)
    assert np.array_equal(L2_a, L2_b)
    assert np.array_equal(L1_a, L2_a)
    assert not np.all(L1_a[support_mask] == -1.0)

def test_random_initialization_still_available():
    (_, support_mask) = make_problem()
    np.random.seed(1)
    (L1_a, L2_a) = initialize_admm_state(3, support_mask, 0, 'random')
    np.random.seed(999)
    (L1_b, L2_b) = initialize_admm_state(3, support_mask, 0, 'random')
    assert not np.array_equal(L1_a, L1_b)
    assert not np.array_equal(L2_a, L2_b)

def test_invalid_initialization_strategy():
    (_, support_mask) = make_problem()
    try:
        initialize_admm_state(3, support_mask, 0, 'bad')
    except ValueError:
        return
    raise AssertionError('Expected ValueError for invalid init_strategy.')

def test_admm_solve_accepts_both_initialization_strategies():
    (Sigma, support_mask) = make_problem()
    for init_strategy in ('halton', 'random'):
        np.random.seed(42)
        (Lambda, omega) = admm_solve(Sigma, support_mask, max_iter=20, max_restarts=2, init_strategy=init_strategy)
        assert np.all(np.isfinite(Lambda))
        assert np.isfinite(omega)
