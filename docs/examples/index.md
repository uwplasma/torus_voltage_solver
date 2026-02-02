# Examples

All runnable scripts live under `examples/` and are split into three tiers:

- `examples/1_simple/`: minimal, analytic, or near-analytic demonstrations
- `examples/2_intermediate/`: optimization workflows and interactive intuition-builders
- `examples/3_advanced/`: VMEC targets, regularization scans, and more research-style scripts

## Outputs (figures + ParaView)

Most scripts write:

- **Figures** into `figures/<example_name>/…`
- **ParaView datasets** into `figures/<example_name>/paraview/…`

The ParaView entry point is usually a single file:

- `figures/<example_name>/paraview/scene.vtm`

ParaView output is enabled by default; pass `--no-paraview` to disable.

Several examples and GUIs also support optionally *adding* an external ideal toroidal field
$B_\phi \propto 1/R$ (for field-line tracing / visualization) via:

- CLI flags like `--Bext0 <Tesla>` (non-GUI scripts), or
- GUI hotkeys (see each GUI page).

### GUIs (VTK) exports

The interactive GUIs export the current state to ParaView when you press:

- `E` (export)

The export path is:

- `paraview/gui_<name>_<timestamp>/scene.vtm`

```{toctree}
:maxdepth: 1

1_simple
2_intermediate
3_advanced
paraview
```
