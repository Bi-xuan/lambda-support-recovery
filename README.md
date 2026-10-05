# lambda-support-recovery

Select the model support of a Lambda matrix from an empirical covariance matrix.
The solver, nested support search, theorem penalty, and Plateau/Plateau_Bootstrap
procedures are adapted from the MS-S research project.

## Installation

Use Python 3.10 or newer. From the repository root:

```bash
python -m pip install .
```

For development, testing, and building:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m build
```

NumPy and SciPy are the runtime dependencies. Plotting packages are not required.

## Main function

```python
from lambda_support_recovery import select_support

support = select_support(sigma_hat, num_samples=N)
```

`sigma_hat` must be a real, finite, symmetric, positive-definite matrix of size
at least 2. `num_samples` is the original integer sample size, at least 2.
The result is an n-by-n Boolean model mask including all diagonal positions.
Permitted coefficients remain in the support even when fitted to zero.
Model dimension D_m is one plus the number of off-diagonal positions.

The defaults estimate a fixed reference as `0.93 * lambda_min(sigma_hat)`,
construct a greedy forward-nested objective/support curve, then select a
dimension with Plateau_Bootstrap. The reference remains fixed; optional
free-omega refinement is disabled. At each dimension, the search chooses the
best one-edge extension of the preceding support.

The default scope permits every directed off-diagonal position. Set
`support_scope="upper"` when an upper-triangular structure is appropriate.
`preselect_edges=[(i, j), ...]` restricts an `"all"` search to specified directed
positions. Indices are zero-based.

## Detailed results

```python
result = select_support(sigma_hat, num_samples=N, return_result=True)

result.support
result.selected_dimension
result.selected_edges
result.fitted_lambda
result.fitted_omega
result.resolved_omega_ref
result.curve.raw_objectives
result.penalty_constants
result.bootstrap_comparisons
```

Curve/result arrays are read-only copies. The mask-only return is a writable
copy. The API does not read or write files and is silent unless a callback
such as `progress=print` is supplied.

## Configurable penalty quantities

By default, **the observed covariance** supplies its matrix norms and
eigenvalue summaries. The Lambda bound in the theorem penalty defaults to
**L = 1**, the noise bound to **r = 1**, and the confidence constant to **xi = 10**.
These are penalty inputs, not additional constraints on ADMM. Data-derived
quantities do not claim to be population upper bounds.

```python
from lambda_support_recovery import PenaltyConfig, select_support

penalty = PenaltyConfig(
    sigma=None,          # observed covariance (default)
    lambda_bound=0.8,    # theorem L; default 1
    noise_bound=1.0,     # theorem r; default 1
    xi=10.0,
)
support = select_support(sigma_hat, num_samples=N, penalty_config=penalty)
```

Set `sigma=1.0` to use the identity as the penalty reference, another positive
scalar for that scalar times the identity, or a positive-definite n-by-n
matrix for a custom reference. Only the penalty changes; fitting uses
`sigma_hat` throughout.

All six numerical summaries can be overridden individually:

```python
penalty = PenaltyConfig(
    lambda_bound=0.8,
    lambda_sum=4.0, lambda_inf_norm=2.0, lambda_2_norm=3.0,
    sigma_fro_norm=6.0, sigma_op_norm=4.0, sigma_trace=10.0,
)
```

The original `lambda_*` quantities summarize **covariance eigenvalues**, not
the fitted Lambda matrix: their defaults are the eigenvalue sum, maximum
absolute value, and Euclidean norm. The `sigma_*` quantities are the reference's
Frobenius norm, operator norm, and trace. `lambda_bound` is the configurable
bound associated with the Lambda model. The theorem requires
`0 < lambda_bound < sqrt(n)`.

The retained penalty is `sqrt(D_m) * (K + sqrt(2 * v * Lm))`. With constant Lm,
changing penalty quantities rescales the penalty axis and need not change the
dimension. Support-count weighting can change the penalty's relative shape.

## Known or free omega

```python
support = select_support(
    sigma_hat, num_samples=N,
    omega_star=known_omega, omega_ref=known_omega, fit_omega_ref=False,
)
```

`omega_star` is optional known-noise provenance; it never generates data or
silently changes `omega_ref`. A numeric reference requires
`fit_omega_ref=False`. The observed fit requires the reference to be at most
the smallest eigenvalue of `sigma_hat`; equality is permitted.

For free-omega fitting, use `omega_ref=None, fit_omega_ref=False`.
`refine_after_fixed_omega=True` performs the support search with a fixed
reference, then refits each chosen mask with free omega. Bootstrap uses the
curve's final settings for all refits.

## Plain Plateau without bootstrap

```python
support = select_support(sigma_hat, num_samples=N, method="plateau", lm_weight=0.1)
```

Plain Plateau defaults to support-count weighting:
`Lm(D_m) = lm_weight * C(M, D_m - 1)`, where M is the number of allowed
off-diagonal positions; the default weight is 0.1. This counts all supports in
the allowed space, not just the extensions searched along the greedy path.

Set `lm_mode="constant"` to use constant Lm (default 1). A constant weight
rescales the penalty axis without changing plateau widths. Ordinary Plateau
preserves MS-S's default selection at twice the geometric center of its widest
bounded plateau. Set `recommendation_factor=1.0` to select at the center.

Plateau_Bootstrap defaults to constant Lm=1 and rejects support-count weighting.
The only selection methods are plain `"plateau"` and `"plateau_bootstrap"`
(also accepted as `"plateau-bootstrap"`). Capitalized names are accepted too.
The low-level numerical entry points in `selection.py` are `select_plateau`
and `select_plateau_bootstrap`.

## Reuse a curve

```python
from lambda_support_recovery import compute_support_curve, select_from_curve

curve = compute_support_curve(sigma_hat, num_samples=N)
bootstrap = select_from_curve(curve, return_result=True)
plateau = select_from_curve(
    curve, method="plateau", lm_weight=0.2,
    penalty_config=PenaltyConfig(lambda_bound=0.8), return_result=True,
)
```

This avoids repeating support reconstruction. Bootstrap still performs its
fixed-mask refits. A recorded original sample size cannot be replaced by a
different value. If the curve was built without `num_samples`, supply it when
selecting.

## Parameters and defaults

Python keywords use lowercase names corresponding to the scripts' uppercase
configuration names.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_restarts` | 10 | Initializations per support and bootstrap refit |
| `omega_star` | None | Optional known-noise provenance |
| `omega_ref`, `fit_omega_ref` | None, True | Estimate and fix reference omega |
| `objective_floor` | 1e-8 | Screening floor; raw gains stay unchanged |
| `lm_weight` | 1 / 0.1 | Constant / support-count weight |
| `lm_mode` | constant / support-count | Bootstrap / plain Plateau default |
| `top_plateaus` | 3 | Bootstrap candidates screened |
| `bootstrap_replicates` | 199 | Draws per comparison |
| `bootstrap_alpha` | 0.05 | Strict rejection threshold |
| `penalty_config` | PenaltyConfig() | Covariance reference and penalty bounds |
| `kappa` | 0.93 | Estimated-reference multiplier |
| `max_iter`, `tol` | 800, 1e-7 | Solver budget and tolerance |
| `beta`, `zero_tol`, `obj_tol` | 1, 1e-5, 1e-8 | ADMM and numerical tolerances |
| `min_omega` | 0 | Lower fitting noise bound |
| `init_strategy` | halton | Random starts also available for plain Plateau |
| `support_scope`, `nested_supports` | all, True | Allowed positions and nesting |
| `preselect_edges` | None | Optional permitted directed edge list |
| `refine_after_fixed_omega` | False | Free-omega final fits |
| `random_seed`, `bootstrap_seed` | 42, 20260913 | Random-start and bootstrap seeds |
| `n_jobs` | 1 | Process workers |
| `recommendation_factor` | 2 | Plain Plateau center multiplier |
| `return_result`, `progress` | False, None | Detailed result and optional callback |

Bootstrap-only settings are unused by plain Plateau; the recommendation factor
is unused by bootstrap. Bootstrap requires nested supports and Halton starts.
In executable scripts using `n_jobs > 1`, invoke the API under an
`if __name__ == "__main__":` guard.

## Bootstrap assumptions and failures

Bootstrap assumes iid zero-mean Gaussian observations and covariance
`X.T @ X / N`. It screens bounded log-width plateaus and tests adjacent
candidates in decreasing dimension. Both original masks are refitted for
every draw; screening and support reconstruction are not repeated.
The original raw objectives are used for gains regardless of the floor.

The p-value is `(1 + exceedances) / (B + 1)`. Only `p < alpha` retains the larger
model and stops; otherwise selection moves to the next comparison and
eventually the smallest candidate. A single candidate needs no comparison.
There is no multiple-testing correction.

Invalid configurations, absent bounded plateaus, ambiguous plain-Plateau ties,
unstable null models, and failed bootstrap refits raise explicit errors.
Invalid curve fits retain Inf objectives and invalid masks; a failed nested
fit blocks later extensions. No alternative method is substituted silently.

## Example and verification

Run `python examples/basic_selection.py` for a small demonstration using
reduced solver budgets and bootstrap draws. The tests adapt MS-S regressions
and compare objectives, masks, penalties, dimensions, and bootstrap gains with
stored synthetic MS-S reference cases. No live MS-S checkout is needed for
testing. The original project remains unchanged.

## Licensing

A license has not yet been selected. Add the chosen LICENSE file and package
license metadata before distributing the implementation for reuse.
