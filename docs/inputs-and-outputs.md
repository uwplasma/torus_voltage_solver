# Inputs and outputs (library + scripts)

This page is a practical reference for **what you can control** (inputs) and **what you get back**
(outputs) when using `torus-solver` as:

1. a Python library (`import torus_solver`), and
2. runnable scripts under `examples/`.

If you are new, start with {doc}`getting-started` first.

## Core library API

Most of the code is built from a small set of composable primitives:

### 1) Winding-surface geometry

**Constructor**

- `torus_solver.make_torus_surface(R0, a, n_theta, n_phi)`

**Inputs**

- `R0` [m]: major radius
- `a` [m]: minor radius
- `n_theta`, `n_phi` [1]: grid sizes for periodic angles $\theta,\phi\in[0,2\pi)$

**Outputs**

- a `TorusSurface` PyTree containing:
  - `theta`, `phi` (grid vectors)
  - `r(θ,φ)` [m], `r_theta`, `r_phi` (tangent vectors)
  - metric/area quantities (`G`, `sqrt_g`, `area_weights`, …)
  - spectral wavenumbers (`k_theta`, `k_phi`) for FFT derivatives

### 2) Electrode model (surface voltage)

**Electrode deposition**

- `torus_solver.deposit_current_sources(surface, theta_src, phi_src, currents, sigma_theta, sigma_phi)`

Inputs:

- `theta_src`, `phi_src` [rad]: electrode locations (arrays of length `Ns`)
- `currents` [A]: injected currents (sources positive, sinks negative)
- `sigma_theta`, `sigma_phi` [rad]: smoothing widths of the periodic deposition kernel

Outputs:

- `s(θ,φ)` [A/m²]: injected current density on the surface grid (area-mean is removed)

**Poisson solve**

- `torus_solver.solve_current_potential(surface, source_density, tol, maxiter, use_preconditioner)`

Inputs:

- `source_density` [A/m²]: right-hand side $b$ in $-\Delta_s V=b$
- `tol`, `maxiter`: CG solve parameters

Outputs:

- `V(θ,φ)` [V] (gauge-fixed to area-mean zero)
- `info`: CG status code (0 means converged)

**Surface current from potential**

- `torus_solver.surface_current_from_potential(surface, V, sigma_s)`

Inputs:

- `sigma_s` [S]: sheet conductivity (uniform scale factor)

Outputs:

- `K(θ,φ)` [A/m]: tangential surface current density

### 3) Current-potential model (REGCOIL-like)

**Surface current from current potential**

- `torus_solver.current_potential.surface_current_from_current_potential_with_net_currents(surface, Phi_sv, net_poloidal_current_A, net_toroidal_current_A)`

Inputs:

- `Phi_sv(θ,φ)` [A]: single-valued (periodic) current potential
- `net_poloidal_current_A` [A], `net_toroidal_current_A` [A]: optional net currents (handled analytically)

Outputs:

- `K(θ,φ)` [A/m]

### 4) Magnetic field evaluation (Biot–Savart)

- `torus_solver.biot_savart_surface(surface, K, eval_points, eps, chunk_size)`

Inputs:

- `K(θ,φ)` [A/m]: surface current density
- `eval_points(N,3)` [m]: points where $\mathbf B$ is evaluated
- `eps` [m]: softening length (numerical regularization near the surface)
- `chunk_size` [1]: trades memory for compute (large `N` → use chunking)

Outputs:

- `B(N,3)` [T]

### 5) Field-line tracing

- `torus_solver.fieldline.trace_field_lines_batch(B_fn, r0, step_size, n_steps, normalize)`

Inputs:

- `B_fn(xyz)` → `B`: a function mapping `(N,3)` points to `(N,3)` fields
- `r0(N,3)` [m]: initial points
- `step_size` [m] (if `normalize=True`): approximate arc-length step
- `n_steps` [1]
- `normalize=True`: trace $\mathbf b=\mathbf B/\|\mathbf B\|$ instead of $\mathbf B$

Outputs:

- `traj(n_steps+1, N, 3)` [m]

### 6) Metrics

- `torus_solver.metrics.bn_over_B(B, normals)` returns $B_n/\|\mathbf B\|$ (dimensionless)
- `torus_solver.metrics.bn_over_B_metrics(B, normals, weights)` returns the field plus RMS and max metrics

## Common CLI inputs (examples)

Scripts are intentionally “verbose drivers”. Most scripts accept:

- geometry:
  - `--R0`, `--a`, `--n-theta`, `--n-phi`
- field-line tracing:
  - `--n-fieldlines`, `--fieldline-steps`, `--fieldline-ds`
  - `--biot-savart-eps`
  - optional background field for tracing:
    - `--Bext0` [T at $R=R_0$]: toroidal $B_\phi\propto 1/R$
    - `--Bpol0` [T at $R=R_0$]: poloidal $B_\theta\propto 1/R$
- outputs:
  - `--outdir`
  - `--no-plots`
  - `--no-paraview`

Advanced scripts add model-specific flags (e.g. VMEC surface sampling, current-potential Fourier cutoffs,
regularization weights, optimizer choices).

## Outputs (figures + ParaView)

Most scripts write:

- **Figures** into `figures/<example_name>/...`
- **ParaView datasets** into `figures/<example_name>/paraview/...`

The ParaView entry point is usually:

- `figures/<example_name>/paraview/scene.vtm`

### What is typically exported

Depending on the script, exports can include:

- `winding_surface_*.vtu` with point data such as `V`, `s`, `K`, `Ktheta`, `Kphi`, `|K|`
- `electrodes.vtu` point cloud with `I_A` and sign
- `fieldlines.vtu` polyline bundle
- extra reference point clouds (axis curves, cut rings, target points, normals)

## GUI inputs and outputs

The VTK GUIs support:

- interactive electrode placement (sources/sinks) and typed numeric inputs
- live field-line updates (Biot–Savart + optional background field)
- `E` to export a timestamped ParaView scene under `paraview/gui_<name>_<timestamp>/scene.vtm`
- `s` to save screenshots under `figures/gui_screenshots/`

See the per-example GUI pages for exact key bindings.

