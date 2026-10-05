"""Objective utilities for evaluating Sigma, Lambda, and omega fits."""

import numpy as np

def frobenius_objective(Sigma, Lambda, omega):
    n = Sigma.shape[0]
    residual = Sigma - Lambda.T @ Sigma @ Lambda - omega * np.eye(n)
    return np.linalg.norm(residual, 'fro') ** 2
