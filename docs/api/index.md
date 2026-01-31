# API reference

The API is intentionally small and “research-friendly”: most routines are plain functions operating on `jax.numpy` arrays and small dataclasses.

```{autosummary}
:toctree: generated
:recursive:

torus_solver.torus
torus_solver.spectral
torus_solver.poisson
torus_solver.sources
torus_solver.current_potential
torus_solver.biot_savart
torus_solver.fields
torus_solver.fieldline
torus_solver.optimize
torus_solver.metrics
torus_solver.targets
torus_solver.surface_ops
torus_solver.vmec
torus_solver.plotting
torus_solver.gui_vtk
```

:::{note}
`torus_solver.gui_vtk` can be imported without VTK installed, but running the GUI requires the optional extra `torus-solver[gui]`.
:::
