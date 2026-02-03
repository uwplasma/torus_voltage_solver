# Applications and workflows

This page connects the models in `torus-solver` to common research workflows in toroidal confinement.

## Tokamak-like vacuum fields (validation and intuition)

Even though this project is motivated by stellarators, a circular torus is also a convenient sandbox for **tokamak-like** vacuum fields.

A canonical reference profile is an ideal toroidal field:

$$
B_\phi(R) = B_0\frac{R_0}{R},
$$

which follows from Ampère’s law when there is a net **poloidal** current linking the torus.

In this repository:

- `torus_solver.fields.ideal_toroidal_field(...)` provides the analytical $1/R$ field.
- `torus_solver.current_potential.surface_current_from_current_potential_with_net_currents(...)` lets you set a net poloidal current $I_{\mathrm{pol}}$ and compute the resulting surface current density.
- `torus_solver.biot_savart.biot_savart_surface(...)` evaluates the field from the surface current so you can compare numerically against the analytical $1/R$ profile.
- `torus_solver.fieldline.trace_field_lines_batch(...)` traces field lines to confirm that trajectories remain well-behaved inside the torus.

This “tokamak mode” is useful for:

- unit/regression tests (known analytical scaling)
- sign-convention checks (direction of $B_\phi$ vs sign of $I_{\mathrm{pol}}$)
- performance profiling (Biot–Savart + tracing in a controlled geometry)

## Stellarator-style design proxy: minimize normalized $B_n/|B|$

For stellarator coil design, a common ingredient is to choose currents on a winding surface so that the **normal component** of the magnetic field on a target surface is small:

$$
\mathrm{BnOverB}(\theta,\phi) = \frac{\mathbf B(\theta,\phi)\cdot\hat{\mathbf n}(\theta,\phi)}{|\mathbf B(\theta,\phi)|}.
$$

The script `examples/inverse_design/optimize_vmec_surface_Bn.py` implements a REGCOIL-like optimization loop:

1. Read a VMEC boundary surface (Fourier coefficients) from a `input.*` file.
2. Scale that target surface to fit safely inside a circular torus (the winding surface).
3. Parameterize the winding-surface current using either:
   - an electrode-driven potential model, or
   - a current potential $\Phi$ (recommended for stringent targets).
4. Minimize a smooth proxy for max-norm, typically a weighted $p$-norm with $p\gtrsim 8$.
5. Validate with field-line tracing and diagnostics such as `max|Bn/B|`.

:::{note}
Driving $B_n/|B|\to 0$ on a single interior surface is a powerful objective, but it does not *by itself* guarantee globally nested flux surfaces. Use field-line tracing and Poincaré plots as sanity checks.
:::

## Connecting to near-axis / quasisymmetry tools

Many near-axis and quasisymmetry workflows (e.g. Garren–Boozer constructions) output either:

- an axis curve + surface shape, or
- a VMEC input file describing a boundary by Fourier series.

This repository currently focuses on the second case: VMEC-style Fourier surfaces.
If you generate a quasisymmetric configuration with a near-axis tool, you can export a VMEC boundary and use it as a target in `torus-solver`.

## What this code is (and is not)

This code is designed to be:

- **differentiable** end-to-end (JAX)
- a **research sandbox** (easy to modify objectives/parameterizations)
- focused on a **circular torus winding surface** (for clarity and analytic checks)

This code is *not yet*:

- a full coil engineering tool (manufacturing constraints, coil thickness, discrete coil sets)
- a plasma-response / equilibrium solver (vacuum field only)
- a general winding-surface geometry solver (non-circular surfaces)

The `Development → Roadmap` page lists concrete extension directions.
