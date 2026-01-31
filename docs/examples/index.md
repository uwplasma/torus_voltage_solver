# Examples and workflows

All runnable scripts live in `examples/`. Most scripts write figures into `examples/figures/` or `figures/…` and print diagnostic summaries to stdout.

## Tokamak-like analytic fields

- `examples/trace_fieldlines_tokamak.py`

Demonstrates field-line tracing in a combined toroidal+poloidal field with approximately $1/R$ scaling.

```bash
python examples/trace_fieldlines_tokamak.py
```

## Winding-surface currents and field lines

- `examples/trace_fieldlines_shell_toroidal_field.py`

Traces field lines from currents on the torus surface that create a toroidal-field-like interior field.

```bash
python examples/trace_fieldlines_shell_toroidal_field.py
```

## Helical fields (toy stellarator-like targets)

- `examples/optimize_helical_field.py`

Optimizes discrete electrodes on the winding surface to reproduce a helical target field along a reference curve.

```bash
python examples/optimize_helical_field.py --n-steps 50
```

## VMEC target surface: minimize normalized $B_n$

- `examples/optimize_vmec_surface_Bn.py`

This script supports two models:

1. `--model current-potential` (recommended for VMEC targets): a REGCOIL-like current potential $\Phi$ on the torus winding surface, optimized with L-BFGS.
2. `--model electrodes`: discrete source/sink electrodes driving a surface potential solve.

Run the REGCOIL-like model:

```bash
python examples/optimize_vmec_surface_Bn.py \
  --model current-potential \
  --vmec-input examples/input.QA_nfp2
```

The script reports:

- `rms(Bn/B)` and `max|Bn/B|` on the target surface, and a target threshold of **0.5%** (i.e. `max|Bn/B| < 5e-3`).
- Optional validation on a finer grid (`--val-surf-n-theta`, `--val-surf-n-phi`).
- Optional field-line tracing diagnostics (`--no-fieldlines` to disable).

## REGCOIL-style regularization scan (L-curve)

- `examples/scan_vmec_surface_regularization.py`

Sweeps the regularization weight on the winding-surface current magnitude and plots the tradeoff between
`max|Bn/B|` on the target surface and `K_rms` on the winding surface.

```bash
python examples/scan_vmec_surface_regularization.py --vmec-input examples/input.QA_nfp2
```

## Interactive GUIs (VTK)

All GUIs require the `vtk` Python package:

```bash
pip install -e '.[gui]'
```

- `examples/gui_optimize_vmec_surface_Bn.py`

Interactive electrode optimization, showing:

- Torus surface scalars (e.g. `|K|`, `V`, `K_theta`, `K_phi`)
- Target VMEC surface colored by normalized `(B·n)/|B|`
- 3D field lines updated in real time

```bash
python examples/gui_optimize_vmec_surface_Bn.py --vmec-input examples/input.QA_nfp2
```

:::{tip}
The GUI is meant for interactive exploration and intuition-building. For reproducible science, prefer scripting runs that log parameters and produce figures.
:::
