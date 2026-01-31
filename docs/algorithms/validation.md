# Validation and regression testing

Scientific computing code needs a “trust story”:

1. **Analytic checks** (things we can compute exactly)
2. **Cross-code checks** (compare to established implementations)
3. **Regression tests** (prevent accidental behavior changes)
4. **Convergence tests** (refine resolution and confirm stability)

## What the test suite covers today

Run:

```bash
pytest -q
```

Key validations include:

- **Current potential secular terms**:
  - For $\Phi = (I_\mathrm{pol}/2\pi)\phi$, the magnitude satisfies $|K| = I_\mathrm{pol}/(2\pi R(\theta))$.
  - For $\Phi = (I_\mathrm{tor}/2\pi)\theta$, the magnitude satisfies $|K| = I_\mathrm{tor}/(2\pi a)$ (constant).
  - Tests: `tests/test_current_potential.py`

- **REGCOIL cross-check (axisymmetric sanity test)**:
  - If the REGCOIL netCDF output exists locally, we compare `max|K|` against `torus-solver` for the same torus and net current.
  - Test: `tests/test_current_potential.py`

- **Divergence-free current potential currents**:
  - The current-potential model should satisfy $\nabla_s\cdot\mathbf K = 0$ (up to discretization error).
  - Test: `tests/test_surface_ops.py`

- **Target-surface fitting sanity**:
  - VMEC surfaces are shifted/scaled so they fit inside the circular torus with a `fit_margin`.
  - Test: `tests/test_targets.py`

- **Normalized normal-field metrics**:
  - Basic sanity tests for `Bn/B`, weighted RMS, and weighted p-norm utilities.
  - Test: `tests/test_metrics.py`

## Recommended additional validations for new research

Before trusting new objectives/parameterizations, consider adding:

- Resolution convergence sweeps for $B_n/|B|$
- Comparisons against analytic interior fields for symmetric current patterns
- Sensitivity checks: gradients vs finite differences for small problems
- Physical unit checks: scaling with $I_\mathrm{pol}$ and geometry

:::{important}
Passing tests does not guarantee the physics model matches a real device. Tests ensure internal consistency and protect against accidental code regressions.
:::
