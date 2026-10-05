# Numerical regression cases adapted from MS-S.
"""Fitted reference omega and the fixed-omega eigenvalue bound."""
import json
import numpy as np
import pytest
from lambda_support_recovery import curve as curve
from lambda_support_recovery.optimizers.support_search import optimize_lambda, resolve_omega_ref
from lambda_support_recovery.selection import FitSettings

def test_fitted_reference_uses_smallest_eigenvalue_and_stays_fixed():
    sigma_hat = np.diag([1.0, 2.0])
    (Lambda, omega, objective) = optimize_lambda(sigma_hat, D_m=1, fit_omega_ref=True, kappa=0.93, max_iter=20, max_restarts=1)
    assert Lambda is not None
    assert omega == pytest.approx(0.93)
    assert np.isfinite(objective)

def test_fixed_reference_can_equal_smallest_eigenvalue_with_preselection():
    Sigma = np.eye(2)
    (Lambda, omega, objective) = optimize_lambda(Sigma, D_m=2, omega_ref=1.0, omega_upper_gap=0.1, preselect_k=1, max_iter=20, max_restarts=1)
    assert Lambda is not None
    assert omega == 1.0
    assert np.isfinite(objective)

@pytest.mark.parametrize('omega_ref, fit_omega_ref, kappa', [(0.5, True, 0.93), (None, True, 0.0), (None, True, 1.1)])
def test_invalid_fitted_reference_configuration(omega_ref, fit_omega_ref, kappa):
    with pytest.raises(ValueError):
        resolve_omega_ref(np.eye(2), omega_ref, fit_omega_ref, kappa)
