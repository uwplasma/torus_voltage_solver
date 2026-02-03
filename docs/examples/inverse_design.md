# Inverse design / optimization examples

These scripts solve inverse problems: choose currents (and sometimes electrode locations) on a
circular torus winding surface to match a target field or to reduce $B_n/|B|$ on a target surface.

All scripts write:

- figures to `figures/<example_name>/…`
- ParaView datasets to `figures/<example_name>/paraview/scene.vtm` (disable with `--no-paraview`)

## Electrode optimization: helical target on-axis

Script:

- `examples/inverse_design/optimize_helical_axis_field.py`

```bash
python examples/inverse_design/optimize_helical_axis_field.py --n-steps 200
```

This example chooses axis points and a simple helical “stellarator-like” target field, then
optimizes discrete source/sink electrodes on the winding surface to match the target.

## Toy near-axis target: bumpy axis + rotating ellipse

Script:

- `examples/inverse_design/optimize_bumpy_axis_rotating_ellipse.py`

```bash
python examples/inverse_design/optimize_bumpy_axis_rotating_ellipse.py --n-steps 120
```

This is a deliberately simplified construction inspired by near-axis intuition:

- prescribe a bumpy axis $R(\phi), Z(\phi)$
- build a discrete Frenet-like frame along the axis
- define evaluation points on a rotating ellipse around the axis
- optimize electrodes so the on-ellipse field matches a prescribed target

## VMEC target: minimize normalized $B_n/|B|$

Script:

- `examples/inverse_design/optimize_vmec_surface_Bn.py`

```bash
python examples/inverse_design/optimize_vmec_surface_Bn.py \
  --model current-potential \
  --vmec-input examples/data/vmec/input.QA_nfp2
```

This optimizes either:

- a REGCOIL-like current potential on the torus surface (`--model current-potential`), or
- discrete electrodes (`--model electrodes`)

to reduce the normalized normal field on a VMEC target surface.

## Regularization scan (REGCOIL-style tradeoff curve)

Script:

- `examples/inverse_design/scan_vmec_surface_regularization.py`

```bash
python examples/inverse_design/scan_vmec_surface_regularization.py \
  --vmec-input examples/data/vmec/input.QA_nfp2
```

This sweeps the current-magnitude regularization weight and produces a tradeoff curve between
field quality (e.g. max|Bn/B|) and current magnitude (e.g. $K_\mathrm{rms}$).

