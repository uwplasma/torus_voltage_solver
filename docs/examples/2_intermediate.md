# 2. Intermediate examples

These scripts introduce **inverse problems**: optimizing currents (and sometimes their locations) on a circular torus winding surface.

They also include interactive GUIs meant for rapid iteration and intuition-building.

## Electrode optimization: helical target on-axis

Script:

- `examples/2_intermediate/optimize_helical_axis_field.py`

This example chooses a set of axis points and a simple helical “stellarator-like” target field, then optimizes discrete source/sink electrodes on the winding surface to match the target.

The objective is a weighted least squares:

$$
L = \\left\\langle \\frac{\\|B(\\mathbf x_i) - B_{\\mathrm{target}}(\\mathbf x_i)\\|^2}{B_{\\mathrm{scale}}^2} \\right\\rangle
\\; + \\; \\text{regularization}.
$$

```bash
python examples/2_intermediate/optimize_helical_axis_field.py --n-steps 200
```

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

## GUI demos

These are “preloaded” starting points:

- `examples/2_intermediate/gui_torus_demo_helical.py`
- `examples/2_intermediate/gui_torus_demo_inboard_cut.py`

They are useful for quickly seeing a nontrivial configuration without manually placing electrodes.

