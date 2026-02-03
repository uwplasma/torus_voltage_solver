# GUI examples (interactive, VTK)

These scripts launch interactive 3D GUIs to place electrodes, change their currents, and see the
resulting surface currents and traced field lines.

Install GUI extras first:

```bash
pip install -e '.[gui]'
```

## Interactive electrode GUI

Script:

- `examples/gui/gui_torus_electrodes_interactive.py`

```bash
python examples/gui/gui_torus_electrodes_interactive.py
```

Key points:

- red electrodes are **sources** (+I), blue are **sinks** (−I)
- press `E` to export the current state to ParaView
- press `B` to toggle adding an external toroidal field ($B_\phi\propto 1/R$) to the traced field lines
- press `P` to toggle adding an external poloidal field ($B_\theta\propto 1/R$) to the traced field lines

## GUI demos (preloaded configurations)

These are “starting points” that preload a configuration so you immediately see non-trivial
currents and field lines:

- `examples/gui/gui_torus_demo_helical.py`
- `examples/gui/gui_torus_demo_inboard_cut.py`

## VMEC optimization GUI

Script:

- `examples/gui/gui_optimize_vmec_surface_Bn.py`

```bash
python examples/gui/gui_optimize_vmec_surface_Bn.py --vmec-input examples/data/vmec/input.QA_nfp2
```

