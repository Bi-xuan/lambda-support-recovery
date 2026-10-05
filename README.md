# lambda-support-recovery

A Python package under development for selecting the model support of Lambda
from an empirical covariance matrix (Sigma_hat).

## Current status

This repository contains packaging metadata, documentation, and empty Python
module placeholders. The numerical implementation and tests have not yet been
added. There is currently no working support-selection function.

The existing MS-S research project remains separate. Its numerical routines will
be adapted into this package in a later implementation stage.

## Planned interface

The main public function will be `select_support`, accepting an empirical
covariance matrix and the original sample size. Its default output will be a
Boolean mask of the selected Lambda model support. Optional detailed results
will contain the selected dimension, objective/support curve, resolved reference
omega, and selection diagnostics.

The Python import package is `lambda_support_recovery`; the distribution name
is `lambda-support-recovery`.

## Planned procedure

1. Construct a nested objective and support curve from the supplied covariance.
2. By default, use `OMEGA_REF=None` and `FIT_OMEGA_REF=True` to derive a fixed
   reference from the empirical covariance.
3. Select a dimension using Plateau_Bootstrap by default.
4. Return the selected model mask, preserving allowed positions even when their
   fitted coefficients are zero.

The original sample size is required for bootstrap calibration and the current
penalty calculation. The model dimension convention is one plus the number of
selected off-diagonal positions; diagonal positions remain in the model.

## Planned configuration

The interface will keep these parameters configurable:

- `MAX_RESTARTS`
- `OMEGA_STAR`
- `OMEGA_REF`
- `FIT_OMEGA_REF`
- `OBJECTIVE_FLOOR`
- `LM_WEIGHT`
- `TOP_PLATEAUS`
- `BOOTSTRAP_REPLICATES`
- `BOOTSTRAP_ALPHA`

Python keyword arguments will use the corresponding lowercase names.

Known omega can be used by setting `OMEGA_REF=OMEGA_STAR` and
`FIT_OMEGA_REF=False`. Plain Plateau selection without bootstrap will also be
available, with explicit penalty weighting. Reference fitting, refinement,
penalty-weight semantics, and solver settings will be finalized and documented
when the numerical API is implemented.

## Repository layout

- `src/lambda_support_recovery/`: empty placeholders for the public API,
  result/configuration types, curve construction, selection, and numerical code.
- `tests/`: guidance for future regression and package-installation checks.
- `examples/`: guidance for future usage examples.
- `pyproject.toml`: build configuration, package metadata, and dependencies.

## Development setup

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

This installs the scaffold; it does not provide support-selection functionality.
Once the implementation and tests are added, run:

```bash
python -m pytest
python -m build
```

No automated tests exist at this stage, so pytest is not yet a validation step.

## Licensing

A license has not yet been selected. Add the chosen LICENSE file and package
license metadata before distributing the implementation for reuse.
