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

result.support                 # Selected support
result.selected_dimension      # D_m = 1 + number of off-diagonal edges
result.selected_edges          # Off-diagonal (i, j) pairs; zero-based
result.fitted_lambda           # Fitted Lambda matrix
result.fitted_omega             # Fixed omega used for fitting
result.curve.raw_objectives     # Unpenalized objective curve
result.bootstrap_comparisons   # Bootstrap results; empty for plain Plateau
```

## Important parameters

| parameter | options | default | instruction |
| --- | --- | --- | --- |
| `method` | `"plateau-bootstrap"`, `"plateau"` | `"plateau-bootstrap"` | `"plateau-bootstrap"` screens the widest plateaus and uses bootstrap tests to compare candidate supports; `"plateau"` selects at twice the geometric center of the widest bounded plateau by default. `"plateau_bootstrap"` is an accepted alias. |
| `support_scope` | `"all"`, `"upper"` | `"all"` | `"all"` allows every directed off-diagonal position; `"upper"` allows only positions above the diagonal. |
| `omega_star` | `None`, nonnegative number | `None` | `None` records no known population noise; a number records the known noise for reference only. To use known noise for fitting, also set `omega_ref` to that value and `fit_omega_ref=False`. |
| `omega_ref` | `None`, nonnegative number | `None` | `None` requires `fit_omega_ref=True` to estimate the noise reference; a number supplies the fixed noise used during all fits and requires `fit_omega_ref=False`. The supplied value must not exceed `lambda_min(sigma_hat)`. |
| `fit_omega_ref` | `True`, `False` | `True` | `True` estimates `omega_ref = kappa * lambda_min(sigma_hat)` once (default `kappa=0.93`) and requires `omega_ref=None`; `False` uses the supplied numeric `omega_ref`. Omega stays fixed during all fits in either case. |
| `lm_mode` | `None`, `"constant"`, `"support-count"` | `None` | `None` chooses `"constant"` for bootstrap and `"support-count"` for plain Plateau. `"constant"` uses $L_m=w$; `"support-count"` uses $L_m(D_m)=w\binom{M}{D_m-1}$, where `w = lm_weight` and $M$ is the number of permitted off-diagonal positions. Bootstrap requires `"constant"`. |
| `lm_weight` | `None`, positive number | `None` | `None` resolves to `1.0` for constant weighting or `0.1` for support-count weighting; a positive number sets the multiplier $w$ in the selected $L_m$ formula. |
| `n_jobs` | Positive integer | `1` | `1` uses one worker process; larger values allow up to that many CPU worker processes. For `n_jobs > 1`, call the package inside `if __name__ == "__main__":` in a Python script. `-1` and `None` are unsupported. |

## Other configurable parameters

All are keywords of `select_support`; only `sigma_hat` and `num_samples` are required.

| Parameter | Default | Purpose |
| --- | --- | --- |
| `nested_supports` | `True` | Greedy nested search; `False` searches all supports per dimension |
| `kappa` | `0.93` | Omega-estimation multiplier |
| `max_restarts` | `10` | Initializations per support |
| `beta` | `1.0` | ADMM penalty parameter |
| `max_iter`, `tol` | `800`, `1e-7` | Iteration limit and convergence tolerance |
| `zero_tol`, `obj_tol` | `1e-5`, `1e-8` | Coefficient threshold and support-comparison tolerance |
| `init_strategy` | `"halton"` | Halton or `"random"` initialization |
| `objective_floor` | `1e-8` | Objective floor for plateau screening |
| `penalty_config` | `None` | Resolves to `PenaltyConfig()` below |
| `top_plateaus` | `3` | Bootstrap candidates |
| `bootstrap_replicates`, `bootstrap_alpha` | `199`, `0.05` | Replicates and rejection threshold |
| `random_seed`, `bootstrap_seed` | `42`, `20260913` | Random-initialization and bootstrap seeds |
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
