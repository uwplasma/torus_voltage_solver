# Examples

Runnable scripts live under `examples/` and are grouped by *what you want to do*:

- `examples/fieldline_tracing/`: trace field lines in analytic and simple Biot–Savart fields
- `examples/validation/`: convergence / analytic checks (build trust in the numerics)
- `examples/inverse_design/`: optimization workflows (inverse problems)
- `examples/gui/`: interactive VTK GUIs
- `examples/performance/`: speed/scaling demos (JAX vs NumPy)
- `examples/data/`: example inputs (e.g. VMEC boundary files)

## Outputs (figures + ParaView)

Most scripts write:

- figures to `figures/<example_name>/…`
- ParaView datasets to `figures/<example_name>/paraview/scene.vtm` (disable with `--no-paraview`)

## Quick start

```bash
python examples/fieldline_tracing/tokamak_like_fieldlines.py
python examples/inverse_design/optimize_helical_axis_field.py --n-steps 50
python examples/inverse_design/scan_vmec_surface_regularization.py --vmec-input examples/data/vmec/input.QA_nfp2
```

