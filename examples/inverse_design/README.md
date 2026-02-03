# Inverse design / optimization

These scripts solve inverse problems on a circular torus winding surface:

- choose electrode sources/sinks to match a target field, or
- choose a current potential to reduce $B_n/|B|$ on a target surface.

Scripts:

- `optimize_helical_axis_field.py`: optimize electrodes to match a helical on-axis target field
- `optimize_bumpy_axis_rotating_ellipse.py`: toy near-axis target (bumpy axis + rotating ellipse)
- `optimize_vmec_surface_Bn.py`: minimize normalized $B_n/|B|$ on a VMEC target surface
- `scan_vmec_surface_regularization.py`: REGCOIL-style regularization scan (“L-curve”)

VMEC input shipped with the repo:

- `examples/data/vmec/input.QA_nfp2`

