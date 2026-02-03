# GUI (interactive VTK)

These scripts launch interactive 3D GUIs (VTK).

Install GUI extras:

```bash
pip install -e '.[gui]'
```

Scripts:

- `gui_torus_electrodes_interactive.py`: add/move/delete electrodes and change currents while seeing field lines
- `gui_torus_demo_helical.py`: preloaded helical electrode pattern demo
- `gui_torus_demo_inboard_cut.py`: toroidal-cut voltage demo + optional added electrodes
- `gui_optimize_vmec_surface_Bn.py`: interactive optimization to reduce $B_n/|B|$ on a VMEC target

Tip: press `E` in the GUI to export a ParaView scene.

