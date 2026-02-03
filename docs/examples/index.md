# Examples

All runnable scripts live under `examples/` and are grouped by *what you want to do*:

- `examples/fieldline_tracing/`: trace field lines in analytic and simple Biot–Savart fields
- `examples/validation/`: convergence / analytic checks (build trust in the numerics)
- `examples/inverse_design/`: optimization workflows (inverse problems)
- `examples/gui/`: interactive VTK GUIs
- `examples/performance/`: speed/scaling demos (JAX vs NumPy)

## Outputs (figures + ParaView)

Most scripts write:

- **Figures** into `figures/<example_name>/…`
- **ParaView datasets** into `figures/<example_name>/paraview/…`

The ParaView entry point is usually a single file:

- `figures/<example_name>/paraview/scene.vtm`

ParaView output is enabled by default; pass `--no-paraview` to disable.

Several examples and GUIs also support optionally *adding* an external background field
that decays like $1/R$ (for field-line tracing / visualization), via:

- CLI flags like `--Bext0 <Tesla>` (toroidal) and `--Bpol0 <Tesla>` (poloidal) in non-GUI scripts, or
- GUI hotkeys (see each GUI page).

### GUIs (VTK) exports

The interactive GUIs export the current state to ParaView when you press:

- `E` (export)

The export path is:

- `paraview/gui_<name>_<timestamp>/scene.vtm`

```{toctree}
:maxdepth: 1

fieldline_tracing
validation
inverse_design
gui
performance
paraview
```
