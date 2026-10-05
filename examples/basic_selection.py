"""Run bootstrap and plain Plateau selection on a small empirical covariance."""
import numpy as np

from lambda_support_recovery import PenaltyConfig, compute_support_curve, select_from_curve


def main():
    rng = np.random.default_rng(7)
    a = rng.normal(size=(4, 4))
    population_covariance = a @ a.T / 4 + np.eye(4)
    num_samples = 1000
    observations = rng.multivariate_normal(np.zeros(4), population_covariance, size=num_samples)
    sigma_hat = observations.T @ observations / num_samples

    curve = compute_support_curve(
        sigma_hat, num_samples=num_samples, support_scope="upper",
        max_restarts=2, max_iter=120,
    )
    # Small fitting budgets and replicate counts keep this example quick.
    # The package defaults are 10 restarts, 800 iterations, and 199 replicates.
    bootstrap = select_from_curve(curve, bootstrap_replicates=39, return_result=True)
    plateau = select_from_curve(
        curve, method="plateau", lm_weight=0.1,
        penalty_config=PenaltyConfig(lambda_bound=1.0), return_result=True,
    )
    for label, result in (("Plateau_Bootstrap", bootstrap), ("Plateau", plateau)):
        print(f"{label}: dimension {result.selected_dimension}, edges {result.selected_edges}")
        print(result.support.astype(int))


if __name__ == "__main__":
    main()
