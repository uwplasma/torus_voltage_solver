# 2. Intermediate examples

These scripts introduce **inverse problems**: optimizing currents (and sometimes their locations) on a circular torus winding surface.

They also include interactive GUIs meant for rapid iteration and intuition-building.

## Electrode optimization: helical target on-axis

Script:

- `examples/2_intermediate/optimize_helical_axis_field.py`

This example chooses a set of axis points and a simple helical “stellarator-like” target field, then optimizes discrete source/sink electrodes on the winding surface to match the target.

### Model and objective

The electrode model solves a surface Poisson problem to obtain the surface potential and current:

$$
-\Delta_s V = \frac{s}{\sigma_s},\qquad
\mathbf K = -\sigma_s \nabla_s V.
$$

The sources/sinks are modeled as smooth periodic kernels that integrate to the injected currents:

$$
s(\theta,\phi)=\sum_i I_i\,w_i(\theta,\phi),\qquad
\iint w_i\,dA = 1.
$$

Given $\mathbf K(\theta,\phi)$, the vacuum magnetic field is computed with Biot–Savart:

$$
\mathbf B(\mathbf x)=\frac{\mu_0}{4\pi}\iint \frac{\mathbf K(\mathbf r')\times (\mathbf x-\mathbf r')}{\|\mathbf x-\mathbf r'\|^3}\,dA'.
$$

The objective is a weighted least squares mismatch to a prescribed target field on a set of axis points $\mathbf x_i$:

$$
L = \left\langle \frac{\|\mathbf B(\mathbf x_i) - \mathbf B_{\mathrm{target}}(\mathbf x_i)\|^2}{B_{\mathrm{scale}}^2} \right\rangle
\;+\; \text{regularization}.
$$

In this example, the target is a simple “helical” pattern:

$$
\mathbf B_{\mathrm{target}}(\phi) =
B_0\,\hat{\mathbf e}_\phi
 + B_1\left[\cos(N_{\mathrm{fp}}\phi)\,\hat{\mathbf e}_R + \sin(N_{\mathrm{fp}}\phi)\,\hat{\mathbf e}_Z\right],
$$

evaluated on the magnetic axis $R=R_0$, $Z=0$.

```bash
python examples/2_intermediate/optimize_helical_axis_field.py --n-steps 200
```

Optional background field (for *field-line tracing only*):

- `--Bext0 <Tesla>` adds an external ideal toroidal field $B_\phi = B_{\mathrm{ext0}} (R_0/R)$ to the traced field lines.
- `--Bpol0 <Tesla>` adds an external tokamak-like poloidal component
  $B_\theta = B_{\mathrm{pol0}} (R_0/R)$ to the traced field lines.

Use `--Bext0 0 --Bpol0 0` to disable.

Outputs include:

- surface maps (`V`, `s`, `|K|`, components)
- optimization history
- 3D field lines
- `figures/optimize_helical_axis_field/paraview/scene.vtm`

## Interactive electrode GUI (VTK)

Script:

- `examples/2_intermediate/gui_torus_electrodes_interactive.py`

This GUI lets you add/move/delete electrodes and change their currents while seeing:

- winding surface scalars (`V`, `s`, `|K|`, …)
- traced field lines (from Biot–Savart)

```bash
pip install -e '.[gui]'
python examples/2_intermediate/gui_torus_electrodes_interactive.py
```

Key points:

- red electrodes are **sources** (+I), blue are **sinks** (−I)
- press `E` to export the current state to `paraview/gui_torus_electrodes_<timestamp>/scene.vtm`
- press `B` to toggle adding an external toroidal field ($B_\phi\propto 1/R$) to the traced field lines
- press `P` to toggle adding an external poloidal field ($B_\theta\propto 1/R$) to the traced field lines
- press `[`/`]` to decrease/increase `Bext0`, and `,`/`.` to decrease/increase `Bpol0`

## GUI demos

These are “preloaded” starting points:

- `examples/2_intermediate/gui_torus_demo_helical.py`
- `examples/2_intermediate/gui_torus_demo_inboard_cut.py`

They are useful for quickly seeing a nontrivial configuration without manually placing electrodes.
