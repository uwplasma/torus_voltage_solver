# Field-line tracing

These scripts focus on tracing field lines and visualizing the resulting geometry/fields.

Start here:

```bash
python examples/fieldline_tracing/tokamak_like_fieldlines.py
```

Scripts:

- `tokamak_like_fieldlines.py`: analytic tokamak-like field with $B_\phi$ and $B_\theta$ scaling like $1/R$
- `external_toroidal_fieldlines.py`: pure external ideal toroidal field $B_\phi \propto 1/R$
- `shell_toroidal_fieldlines.py`: uniform poloidal surface current sheet → toroidal interior field
- `inboard_cut_toroidal_field.py`: toroidal cut voltage (multi-valued potential) → poloidal current → toroidal field

Outputs:

- figures to `figures/<example_name>/…`
- ParaView scene to `figures/<example_name>/paraview/scene.vtm` (disable with `--no-paraview`)

