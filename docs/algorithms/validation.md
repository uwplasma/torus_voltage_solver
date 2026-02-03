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

- **Analytic interior-field checks for symmetric current patterns**:
  - Net poloidal current on the winding surface should produce an approximately toroidal field with
    $B_\phi(R) \approx \mu_0 I_\mathrm{pol}/(2\pi R)$ (Ampère’s law).
  - Scaling check: for fixed $I_\mathrm{pol}$, $B_\phi(R_0)\propto 1/R_0$.
  - Test: `tests/test_validation_physics.py`

- **Resolution convergence for $B_n/|B|$**:
  - For an axisymmetric toroidal field, $B_n/|B|$ should be ~0 on an interior torus and should
    decrease as the winding-surface resolution increases.
  - Unit test: `tests/test_validation_physics.py`
  - Example sweep + plots: `examples/validation/convergence_bn_over_B_toroidal_current.py`

- **Sensitivity checks (autodiff gradients vs finite differences)**:
  - Small problems validate JAX gradients against central finite differences for both:
    - electrode-current parameters, and
    - current-potential coefficients.
  - For robustness, the electrode sensitivity test uses an *exact* linear solve for the Poisson system
    at tiny resolution (finite differences can be unstable if solver stopping criteria change).
  - Test: `tests/test_sensitivity_gradients.py`

## Recommended additional validations for new research

Before trusting new objectives/parameterizations, consider adding:

- Cross-code comparisons against REGCOIL for 3D configurations (beyond axisymmetry)
- Cross-code comparisons against other Biot–Savart implementations (e.g. SIMSOPT)
- Convergence studies for optimization results (not just forward fields)
- Stronger unit tests for GUI-exported ParaView scenes (sanity-check scalars and naming)

:::{important}
Passing tests does not guarantee the physics model matches a real device. Tests ensure internal consistency and protect against accidental code regressions.
:::
