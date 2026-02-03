# Testing and validation

This project treats testing as part of the “trust story” for scientific code:

- **unit tests**: protect low-level operators and invariants
- **regression tests**: prevent accidental numerical drift
- **analytic checks**: compare to closed-form results when available
- **cross-checks**: compare to established tools when possible (optional, local)

For the broader philosophy and the current validation inventory, see {doc}`../algorithms/validation`.

## Run the test suite

From the repo root:

```bash
pytest -q
```

Notes:

- The tests enable double precision (`jax_enable_x64=True`) for tighter comparisons.
- Some optional cross-checks are skipped if dependencies or data are missing (e.g. `netCDF4` or a local REGCOIL output file).

## What tests cover

The test suite includes, among others:

- geometry/metric identities on the circular torus
- spectral Laplace–Beltrami identities for known Fourier modes
- Poisson solver inversion checks (up to gauge)
- Biot–Savart checks:
  - on-axis field of a circular loop (analytic)
  - uniform poloidal current sheet producing an approximately toroidal $1/R$ field
  - chunking invariance (chunked evaluation matches direct evaluation)
- additional physics validations:
  - Ampère-law $B_\phi(R)\approx \mu_0 I_\mathrm{pol}/(2\pi R)$ check for net poloidal current
  - $B_\phi(R_0)\propto 1/R_0$ scaling check
  - convergence of max|$B_n/|B|$| on an interior torus for an axisymmetric toroidal field
- sensitivity/gradient checks:
  - autodiff gradients vs central finite differences on tiny problems (uses an exact Poisson solve)
- electrode model scaling consistency with the documented PDE:
  - for uniform $\sigma_s$, $\mathbf K$ is invariant and $\sigma_s V$ is invariant
- field-line tracing consistency:
  - pure toroidal field traces a circle (constant $R,Z$)
  - batch tracer matches vmap tracer (up to array layout)

Browse:

- `tests/test_torus_solver.py`
- `tests/test_current_potential.py`
- `tests/test_metrics.py`
- `tests/test_targets.py`
- `tests/test_paraview.py`
- `tests/test_validation_physics.py`
- `tests/test_sensitivity_gradients.py`

## Build documentation locally (recommended)

To ensure the docs build without warnings:

```bash
python -m sphinx -b html -E docs docs/_build/html -W
```

Then open:

- `docs/_build/html/index.html`

## Benchmarks (not tests)

Benchmarks are small scripts intended to sanity-check performance after refactors:

```bash
python benchmarks/bench_forward_and_grad.py
python benchmarks/bench_fieldline.py
```

They print compile time and steady-state runtime for representative kernels.
