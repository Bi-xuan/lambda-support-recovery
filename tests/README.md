# Tests

Run `python -m pytest` after installing the development dependencies.

Numerical regressions are adapted from MS-S for the ADMM solver, support
enumeration, nested recovery, fitted references, theorem penalty, dimension
paths, and fixed-mask bootstrap calibration. Public API tests cover defaults,
known noise, cached selection, penalty overrides, validation, original sample
size, and preservation of selected model masks.

`test_reference_equivalence.py` compares real numerical fits and bootstrap
results with the small reference cases in `fixtures/`. No live MS-S checkout
is needed.
