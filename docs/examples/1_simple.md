# 1. Simple examples

These scripts are designed to be the fastest way to build intuition about:

- the coordinate system and geometry on a circular torus
- the meaning of field-line tracing and “pitch”
- how winding-surface currents produce interior fields (Biot–Savart)
- how this repo exports results to ParaView

## Tokamak-like analytic field lines

Script:

- `examples/1_simple/tokamak_like_fieldlines.py`

This traces field lines in an **analytic** tokamak-like field with approximate $1/R$ scaling. It is a good starting point to understand the field-line tracer without introducing Biot–Savart integrals.

```bash
python examples/1_simple/tokamak_like_fieldlines.py
```

Outputs:

- `figures/tokamak_like_fieldlines/traj/fieldline_3d.png`
- `figures/tokamak_like_fieldlines/paraview/scene.vtm`

## External ideal toroidal field lines (B ~ 1/R)

Script:

- `examples/1_simple/external_toroidal_fieldlines.py`

This traces multiple field lines in a *pure* external ideal toroidal field:

$$
\mathbf B_\mathrm{ext}(R) = B_0\frac{R_0}{R}\,\hat{\mathbf e}_\phi,
$$

so the exact field lines should be circular rings at constant $(R,Z)$.

```bash
python examples/1_simple/external_toroidal_fieldlines.py
```

Key inputs:

- `--B0` (Tesla): sets the background magnitude at `R=R0`.
- `--R0`, `--rho`, `--n-lines`, `--fieldline-steps`, `--fieldline-ds`.

Outputs:

- `figures/external_toroidal_fieldlines/fieldlines/fieldlines_3d.png`
- `figures/external_toroidal_fieldlines/field/Bmag_vs_R.png`
- `figures/external_toroidal_fieldlines/paraview/scene.vtm`

## Shell current sheet → toroidal field

Script:

- `examples/1_simple/shell_toroidal_fieldlines.py`

This prescribes a **uniform poloidal surface current density** $K_\theta$ on the winding surface. In the thin-shell limit, this produces an approximately toroidal field

$$
B_\phi(R) \approx \mu_0 K_\theta \frac{R_0}{R}.
$$

```bash
python examples/1_simple/shell_toroidal_fieldlines.py
```

Outputs include:

- Biot–Savart vs analytic comparisons along a traced line
- `figures/shell_toroidal_fieldlines/paraview/scene.vtm`

## Inboard cut voltage → poloidal current → toroidal field

Script:

- `examples/1_simple/inboard_cut_toroidal_field.py`

A single-valued scalar potential cannot drive a net poloidal loop current on a torus. This script introduces a **toroidal cut**, allowing a **multi-valued** potential with a prescribed voltage drop $V_\mathrm{cut}$ across the cut.

In the axisymmetric (no-$\phi$-dependence) cut-driven case with $s=0$,

$$
\partial_\theta V = \frac{C}{R(\theta)}, \qquad
V_\mathrm{cut} = \oint \partial_\theta V\, d\theta,
$$

which fixes the constant $C$.

```bash
python examples/1_simple/inboard_cut_toroidal_field.py --trace
```

Outputs include:

- a visualization of the cut (where the potential jumps)
- midplane $B_\phi(R)$ vs $1/R$ comparison
- `figures/inboard_cut_toroidal_field/paraview/scene.vtm`

## JAX vs NumPy: Biot–Savart timing

Script:

- `examples/1_simple/jax_vs_numpy_biot_savart_speed.py`

This is a “why JAX?” example. It compares:

- a simple NumPy implementation that loops over evaluation points, and
- the vectorized JAX implementation (compiled with `jax.jit`).

```bash
python examples/1_simple/jax_vs_numpy_biot_savart_speed.py --n-eval 512 --repeat 20
```

Outputs include:

- a timing figure
- `figures/jax_vs_numpy_biot_savart_speed/paraview/scene.vtm`
