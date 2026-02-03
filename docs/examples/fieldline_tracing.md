# Field-line tracing examples

These scripts focus on **field-line tracing** and basic 3D visualization, without requiring an
optimization loop.

All scripts write:

- figures to `figures/<example_name>/…`
- ParaView datasets to `figures/<example_name>/paraview/scene.vtm` (disable with `--no-paraview`)

## Tokamak-like analytic field ($B_\phi$ and $B_\theta$ ~ 1/R)

Script:

- `examples/fieldline_tracing/tokamak_like_fieldlines.py`

```bash
python examples/fieldline_tracing/tokamak_like_fieldlines.py
```

This is the best “first run” for understanding the tracer, since the field is analytic (no
Biot–Savart integral).

## External ideal toroidal field ($B_\phi \propto 1/R$)

Script:

- `examples/fieldline_tracing/external_toroidal_fieldlines.py`

```bash
python examples/fieldline_tracing/external_toroidal_fieldlines.py
```

For a purely toroidal field, field lines should be circles at constant $(R,Z)$.

## Shell current sheet → toroidal field

Script:

- `examples/fieldline_tracing/shell_toroidal_fieldlines.py`

```bash
python examples/fieldline_tracing/shell_toroidal_fieldlines.py
```

This prescribes a uniform poloidal surface current density and compares Biot–Savart against the
ideal $1/R$ trend.

## Inboard cut voltage → poloidal current → toroidal field

Script:

- `examples/fieldline_tracing/inboard_cut_toroidal_field.py`

```bash
python examples/fieldline_tracing/inboard_cut_toroidal_field.py --trace
```

This introduces a **toroidal cut** so the potential can be multi-valued, producing a net poloidal
surface current (and hence a toroidal interior field).

