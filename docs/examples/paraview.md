# ParaView outputs

This project can export results to **VTK XML** formats so you can inspect them in **ParaView** (or any VTK-capable tool) without running the interactive GUI.

## File formats

We use a minimal, dependency-free writer (`torus_solver.paraview`) that produces **ASCII** VTK XML files:

- `.vtu`: `UnstructuredGrid` (used for winding-surface meshes, point clouds, and polylines)
- `.vtm`: `vtkMultiBlockDataSet` (a lightweight “scene” file that references multiple `.vtu` blocks)

In most runs, you only need to open:

- `scene.vtm`

## What gets exported

Exact content depends on the script, but common blocks include:

- `winding_surface*.vtu`
  - Quad mesh of the circular torus winding surface
  - Typical point data:
    - `V`: surface electric potential (electrode model)
    - `s`: injected current density (electrode model)
    - `K`: surface current density vector (A/m)
    - `Ktheta`, `Kphi`, `|K|`: decomposed components/magnitude
    - `Phi`: current potential (current-potential / REGCOIL-like model)
- `electrodes*.vtu`
  - Point cloud of electrode locations on the winding surface
  - Typical point data:
    - `I_A`: projected electrode currents (A) with net current constrained to zero
    - `sign_I`: `sign(I_A)` for quick thresholding/selection
- `target_points.vtu`
  - Point cloud of target surface points (e.g. a VMEC boundary)
  - Typical point data:
    - `Bn_over_B`: normalized normal field on the target surface
    - `n_hat`: unit normals
    - `weight`: area weights (discrete surface Jacobian)
- `fieldlines*.vtu`
  - Polyline cells representing traced field lines
  - Often includes `line_id` and `s_index` as point data for coloring
  - Some examples optionally superpose an external ideal toroidal field $B_\phi\propto 1/R$ in the tracer.
    In that case, the exported field lines correspond to the *total* field used for tracing.

## Recommended ParaView workflow

1. Open `scene.vtm`
2. For surfaces:
   - Use **Color By** → `|K|`, `V`, or `Bn_over_B`
   - Add **Contour** filters to visualize level sets
3. For field lines:
   - Use **Tube** to get publication-quality 3D lines
   - Color by `line_id` or by a scalar you compute in ParaView

## Notes

- Outputs are intended to be **lightweight, portable, and scriptable**.
- The interactive GUIs use VTK for rendering; the exporters do **not** require `vtk` to be installed.
