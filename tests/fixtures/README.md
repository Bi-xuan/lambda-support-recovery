# MS-S numerical reference cases

Fixtures were generated from the unmodified MS-S numerical code at commit
`1a6779c6ff4828d6801789cf7484af8ab08a9ba7`.

The covariance is `A @ A.T / 4 + I`, where A is a 4-by-4 standard-normal draw
from NumPy's default generator with seed 7. The directed case uses its leading
3-by-3 block. Fits use 120 iterations, tolerance 1e-7, two Halton restarts,
nested supports, and otherwise the original curve routine's defaults.

- `estimated.npz`: reference estimated with kappa=0.93. Also records observed-
  covariance penalties, plain Plateau with support-count weight 0.1, and
  bootstrap with N=1000, 19 draws, alpha=0.05, and seed 20260913.
- `known.npz`: fixed reference 0.4.
- `free.npz`: free-omega fitting.
- `refined.npz`: reference 0.4 for support search, then free-omega refinement.
- `directed.npz`: full directed scope with estimated reference.

The other four-variable cases use strictly upper-triangular off-diagonal
positions. These are synthetic results, not user data.
