# Numerics and units

This page summarizes the **physical units** and the **numerical discretizations** used in `torus-solver`.
It is intentionally short; each model page contains the governing equations and links to the implementation.

## Coordinate system and base units

- Lengths are in **meters**: $R_0$ [m], $a$ [m].
- Angles $(\theta, \phi)$ are **dimensionless** and lie on $[0,2\pi)$.
- Cartesian points $\mathbf x = (x,y,z)$ are in **meters**.

## Surface quantities and units

### Model A (electrodes / surface voltage)

- Surface potential: $V(\theta,\phi)$ [V]
- Sheet conductivity (uniform): $\sigma_s$ [S] (Siemens)
- Surface current density: $\mathbf K(\theta,\phi)$ [A/m]
- Injected current density: $s(\theta,\phi)$ [A/m$^2$]
- Electrode parameters:
  - electrode location $(\theta_i,\phi_i)$ [rad]
  - injected current $I_i$ [A]

Governing equations implemented:

$$
\\mathbf K = -\\sigma_s \\nabla_s V,\\qquad
\\nabla_s\\cdot\\mathbf K = -s
\\;\\Rightarrow\\;
-\\Delta_s V = \\frac{s}{\\sigma_s}.
$$

Implementation note:
`solve_current_potential(...)` solves the unit-conductivity form $-\\Delta_s V=b$ (with gauge fixing),
so the electrode model passes `b = s/σ_s` and then uses `K = -σ_s ∇_s V`.

For **uniform** $\sigma_s$, $\mathbf K$ is (up to numerical error) invariant to $\sigma_s$; changing
`sigma_s` mainly rescales $V$.

### Model B (current potential / REGCOIL-like)

- Current potential: $\Phi(\theta,\phi)$ [A]
- Surface current density: $\mathbf K = \\hat{\\mathbf n}\\times\\nabla_s\\Phi$ [A/m]
- Optional net currents: $I_{\\mathrm{pol}}, I_{\\mathrm{tor}}$ [A]

In the VMEC example, the optimized Fourier coefficients are dimensionless and are scaled by `Phi_scale`
to give $\Phi$ in amperes.

## Magnetic field and normalization

- Magnetic field: $\mathbf B(\mathbf x)$ [T]
- Biot–Savart:
  - $\mu_0 = 4\\pi\\times 10^{-7}$ [H/m]
  - inputs: $\mathbf K$ [A/m]
  - outputs: $\mathbf B$ [T]

Throughout the repository we report and optimize the **normalized normal field**

$$
\\frac{B_n}{|B|}
= \\frac{\\mathbf B\\cdot\\hat{\\mathbf n}}{\\|\\mathbf B\\|},
$$

which is **dimensionless**.

## Discretization choices

- The winding surface is a uniform periodic grid in $(\\theta,\\phi)$.
- Surface derivatives use **spectral (FFT) differentiation** in both angles.
- The surface Poisson solve uses **conjugate gradients** (`jax.scipy.sparse.linalg.cg`) with:
  - area-mean gauge fix for $V$
  - optional constant-coefficient spectral preconditioner (`use_preconditioner=True`)
- Field-line tracing uses fixed-step **RK4** with `lax.scan`:
  - If `normalize=True`, the ODE uses $\\mathbf b=\\mathbf B/\\|\\mathbf B\\|$ so `step_size`
    is approximately an arc-length step in meters.

## Practical numerics tips

- Prefer `jax_enable_x64=True` for tight physics comparisons and regression tests.
- Expect a one-time JIT “compile” cost on first call; subsequent calls are fast.
- For large evaluation grids, `biot_savart_surface(..., chunk_size=...)` avoids allocating a huge
  `(N_eval, N_surface, 3)` tensor at once.

