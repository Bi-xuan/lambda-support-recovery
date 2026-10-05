# lambda-support-recovery

Recover the model support of $\Lambda$ from an empirical covariance $\hat{\Sigma}$.

## Simplest usage

Python 3.10+ is required. NumPy and SciPy are installed automatically.

```bash
python -m pip install "git+https://github.com/Bi-xuan/lambda-support-recovery.git"
```

```python
from lambda_support_recovery import select_support

support = select_support(sigma_hat, num_samples=N)
```

Pass a real, finite, symmetric, positive-definite covariance of size at least 2
and its original sample size `N >= 2`. Defaults use a nested support search,
an estimated fixed omega, and Plateau_Bootstrap.

## Results

The default output is an $n\times n$ Boolean support mask including the diagonal.
For fitted parameters and selection details:

```python
result = select_support(sigma_hat, num_samples=N, return_result=True)

result.support                 # Selected mask
result.selected_dimension      # D_m = 1 + number of off-diagonal edges
result.selected_edges          # Off-diagonal (i, j) pairs; zero-based
result.fitted_lambda           # Fitted Lambda matrix
result.fitted_omega             # Fixed omega used for fitting
result.curve.raw_objectives     # Objective curve
result.bootstrap_comparisons   # Bootstrap results; empty for plain Plateau
```

## Personalization

Add these keywords to `select_support(sigma_hat, num_samples=N, ...)`:

| Choice | Keywords |
| --- | --- |
| All directed positions / upper-triangular positions | `support_scope="all"` / `support_scope="upper"` |
| Known omega | `omega_star=known_omega, omega_ref=known_omega, fit_omega_ref=False` |
| Unknown omega | No change: `fit_omega_ref=True` estimates `omega_ref = kappa * lambda_min(sigma_hat)` once |
| Constant $L_m$ | `lm_mode="constant", lm_weight=1.0`, with `method="plateau"` or `method="plateau_bootstrap"` |
| Dimension-dependent $L_m$ | `method="plateau", lm_mode="support-count", lm_weight=0.1` |
| Parallel execution | `n_jobs=4` for up to four CPU worker processes |

Omega stays fixed during all fits. A known reference must not exceed
`lambda_min(sigma_hat)`. Support-count weighting uses
$L_m(D_m)=w\binom{M}{D_m-1}$, where `w = lm_weight` and $M$ is the number of
permitted off-diagonal positions.

For `n_jobs > 1`, call the package inside `if __name__ == "__main__":` in a
Python script. Use a positive integer; `-1` and `None` are unsupported.

## Configurable parameters

All are keywords of `select_support`; only `sigma_hat` and `num_samples` are required.

| Parameter | Default | Purpose |
| --- | --- | --- |
| `method` | `"plateau-bootstrap"` | Bootstrap or plain `"plateau"`; underscore alias accepted |
| `support_scope` | `"all"` | Directed or `"upper"` positions |
| `nested_supports` | `True` | Greedy nested search; `False` searches all supports per dimension |
| `omega_star` | `None` | Record known noise |
| `omega_ref`, `fit_omega_ref` | `None`, `True` | Supply or estimate fixed noise |
| `kappa` | `0.93` | Omega-estimation multiplier |
| `max_restarts` | `10` | Initializations per support |
| `beta` | `1.0` | ADMM penalty parameter |
| `max_iter`, `tol` | `800`, `1e-7` | Iteration limit and convergence tolerance |
| `zero_tol`, `obj_tol` | `1e-5`, `1e-8` | Coefficient threshold and support-comparison tolerance |
| `init_strategy` | `"halton"` | Halton or `"random"` initialization |
| `objective_floor` | `1e-8` | Objective floor for plateau screening |
| `lm_mode` | `None` | Resolves to constant for bootstrap, support-count for Plateau |
| `lm_weight` | `None` | Resolves to 1 for constant, 0.1 for support-count |
| `penalty_config` | `None` | Resolves to `PenaltyConfig()` below |
| `top_plateaus` | `3` | Bootstrap candidates |
| `bootstrap_replicates`, `bootstrap_alpha` | `199`, `0.05` | Replicates and rejection threshold |
| `random_seed`, `bootstrap_seed` | `42`, `20260913` | Random-initialization and bootstrap seeds |
| `n_jobs` | `1` | CPU worker processes |
| `recommendation_factor` | `2.0` | Plain Plateau: multiplier of the widest plateau's geometric center; use 1 to select at the center |
| `return_result`, `progress` | `False`, `None` | Detailed output and optional callback, e.g. `print` |

Bootstrap requires `nested_supports=True`, `init_strategy="halton"`, and constant
$L_m$. Bootstrap settings apply only to bootstrap; `recommendation_factor` applies
only to plain Plateau.

Configure penalty references with
`penalty_config=PenaltyConfig(...)`, importing `PenaltyConfig` from the package:

| `PenaltyConfig` field | Default | Purpose |
| --- | --- | --- |
| `sigma` | `None` | Observed covariance; alternatively a covariance matrix or positive scalar times identity |
| `lambda_bound`, `noise_bound`, `xi` | `1.0`, `1.0`, `10.0` | Theorem constants $L$, $r$, $\xi$ |
| `lambda_sum`, `lambda_inf_norm`, `lambda_2_norm` | `None` | Derive covariance eigenvalue sum, maximum absolute value, and Euclidean norm; override individually |
| `sigma_fro_norm`, `sigma_op_norm`, `sigma_trace` | `None` | Derive covariance Frobenius norm, operator norm, and trace; override individually |

The `lambda_*` summaries refer to covariance eigenvalues. `lambda_bound` is the
Lambda bound and must satisfy $0<L<\sqrt{n}$.

## Non-configurable hyperparameters

These numerical settings and algorithm rules are fixed internally:

| Setting | Value / rule |
| --- | --- |
| Covariance symmetry tolerance | `rtol=1e-10`, `atol=1e-12` |
| Plain Plateau width-tie tolerance | `rtol=1e-12`, `atol=1e-12`; tied maxima raise an error |
| Bootstrap refit-objective tolerance | `rtol=1e-5`, `atol=1e-10` |
| Plateau ranking | Largest bounded log width, $\log(c_{right}/c_{left})$ |
| Bootstrap sampling | Zero-mean iid Gaussian model; $\hat{\Sigma}=X^TX/N$ |
| Bootstrap null stability | Spectral radius of Lambda below 1 and positive omega |
| Bootstrap p-value | $(1+\text{exceedances})/(B+1)$; retain the larger model only when $p<\alpha$ |
