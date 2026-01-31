from __future__ import annotations

"""Lightweight ParaView exporters (VTK XML) with zero non-standard dependencies.

This module writes VTK XML files in ASCII so users can inspect results in ParaView
without requiring the heavyweight `vtk` Python wheels (the interactive GUI uses
VTK, but the solver + exporters do not).

We primarily use:
  - `.vtu` (UnstructuredGrid) for surfaces, point clouds, and field lines.
  - `.vtm` (vtkMultiBlockDataSet) as a convenience wrapper to load multiple `.vtu`
    files at once.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np

from .torus import TorusSurface


# VTK cell type IDs (subset).
VTK_VERTEX = 1
VTK_LINE = 3
VTK_POLY_LINE = 4
VTK_TRIANGLE = 5
VTK_QUAD = 9


@dataclass(frozen=True)
class VTKUnstructuredGrid:
    points: np.ndarray  # (N,3)
    cells: np.ndarray  # (M,k) (uniform k per cell)
    cell_type: int  # VTK_* constant above
    point_data: dict[str, np.ndarray] | None = None
    cell_data: dict[str, np.ndarray] | None = None


def _ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _as_np(a) -> np.ndarray:
    return np.asarray(a)


def _vtk_scalar_type(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    if arr.dtype.kind in ("f", "c"):
        return "Float64"
    if arr.dtype.kind in ("i", "u", "b"):
        if arr.dtype.kind == "u" and arr.dtype.itemsize == 1:
            return "UInt8"
        return "Int32"
    raise TypeError(f"Unsupported dtype for VTK export: {arr.dtype}")


def _coerce_point_array(arr: np.ndarray, *, n_points: int, name: str) -> tuple[np.ndarray, int]:
    """Return (values, n_components) suitable for a VTK PointData DataArray."""
    arr = np.asarray(arr)
    if arr.shape == (n_points,):
        return arr.astype(np.float64, copy=False), 1
    if arr.ndim == 2 and arr.shape[0] == n_points:
        return arr.astype(np.float64, copy=False), int(arr.shape[1])
    raise ValueError(
        f"PointData array '{name}' has shape {arr.shape}, expected ({n_points},) or ({n_points},C)."
    )


def _coerce_cell_array(arr: np.ndarray, *, n_cells: int, name: str) -> tuple[np.ndarray, int]:
    """Return (values, n_components) suitable for a VTK CellData DataArray."""
    arr = np.asarray(arr)
    if arr.shape == (n_cells,):
        return arr.astype(np.float64, copy=False), 1
    if arr.ndim == 2 and arr.shape[0] == n_cells:
        return arr.astype(np.float64, copy=False), int(arr.shape[1])
    raise ValueError(
        f"CellData array '{name}' has shape {arr.shape}, expected ({n_cells},) or ({n_cells},C)."
    )


def _write_data_array_ascii(
    f,
    *,
    arr: np.ndarray,
    vtk_type: str,
    name: str | None = None,
    n_components: int | None = None,
    indent: str = "        ",
    fmt: str | None = None,
) -> None:
    arr = np.asarray(arr)
    attrs = [f'type="{vtk_type}"', 'format="ascii"']
    if name is not None:
        attrs.append(f'Name="{name}"')
    if n_components is not None and int(n_components) != 1:
        attrs.append(f'NumberOfComponents="{int(n_components)}"')
    f.write(f"{indent}<DataArray {' '.join(attrs)}>\n")

    if fmt is None:
        if vtk_type in ("Int32", "UInt8"):
            fmt = "%d"
        else:
            fmt = "%.17g"

    # VTK expects whitespace-separated values. One tuple per line improves readability.
    if arr.ndim == 1:
        data = arr.reshape(-1, 1)
    else:
        data = arr.reshape(arr.shape[0], -1)

    np.savetxt(f, data, fmt=fmt)
    f.write(f"{indent}</DataArray>\n")


def write_vtu(path: str | Path, grid: VTKUnstructuredGrid) -> Path:
    """Write a `.vtu` UnstructuredGrid in ASCII."""
    path = _ensure_parent(path)

    pts = np.asarray(grid.points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {pts.shape}")

    cells = np.asarray(grid.cells, dtype=np.int32)
    if cells.ndim != 2 or cells.size == 0:
        raise ValueError(f"cells must have shape (M,k), got {cells.shape}")

    n_points = int(pts.shape[0])
    n_cells = int(cells.shape[0])
    k = int(cells.shape[1])

    connectivity = cells.reshape(-1).astype(np.int32, copy=False)
    offsets = (np.arange(1, n_cells + 1, dtype=np.int32) * k).astype(np.int32, copy=False)
    types = np.full((n_cells,), int(grid.cell_type), dtype=np.uint8)

    point_data = grid.point_data or {}
    cell_data = grid.cell_data or {}

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{n_points}" NumberOfCells="{n_cells}">\n')

        if point_data:
            f.write("      <PointData>\n")
            for name, arr in point_data.items():
                values, n_comp = _coerce_point_array(arr, n_points=n_points, name=name)
                _write_data_array_ascii(
                    f,
                    arr=values,
                    vtk_type=_vtk_scalar_type(values),
                    name=name,
                    n_components=n_comp,
                    indent="        ",
                )
            f.write("      </PointData>\n")
        else:
            f.write("      <PointData/>\n")

        if cell_data:
            f.write("      <CellData>\n")
            for name, arr in cell_data.items():
                values, n_comp = _coerce_cell_array(arr, n_cells=n_cells, name=name)
                _write_data_array_ascii(
                    f,
                    arr=values,
                    vtk_type=_vtk_scalar_type(values),
                    name=name,
                    n_components=n_comp,
                    indent="        ",
                )
            f.write("      </CellData>\n")
        else:
            f.write("      <CellData/>\n")

        f.write("      <Points>\n")
        _write_data_array_ascii(
            f,
            arr=pts,
            vtk_type="Float64",
            name=None,
            n_components=3,
            indent="        ",
            fmt="%.17g",
        )
        f.write("      </Points>\n")

        f.write("      <Cells>\n")
        _write_data_array_ascii(
            f,
            arr=connectivity,
            vtk_type="Int32",
            name="connectivity",
            n_components=1,
            indent="        ",
            fmt="%d",
        )
        _write_data_array_ascii(
            f,
            arr=offsets,
            vtk_type="Int32",
            name="offsets",
            n_components=1,
            indent="        ",
            fmt="%d",
        )
        _write_data_array_ascii(
            f,
            arr=types,
            vtk_type="UInt8",
            name="types",
            n_components=1,
            indent="        ",
            fmt="%d",
        )
        f.write("      </Cells>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")

    return path


def write_vtm(path: str | Path, blocks: Mapping[str, str | Path]) -> Path:
    """Write a `.vtm` multi-block file referencing other VTK datasets."""
    path = _ensure_parent(path)
    base = path.parent

    with open(path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="vtkMultiBlockDataSet" version="1.0" byte_order="LittleEndian">\n')
        f.write("  <vtkMultiBlockDataSet>\n")
        for i, (name, file) in enumerate(blocks.items()):
            rel = Path(file)
            if rel.is_absolute():
                rel = rel.relative_to(base)
            f.write(f'    <Block index="{i}" name="{name}">\n')
            f.write(f'      <DataSet index="0" file="{rel.as_posix()}"/>\n')
            f.write("    </Block>\n")
        f.write("  </vtkMultiBlockDataSet>\n")
        f.write("</VTKFile>\n")

    return path


def torus_surface_to_vtu(
    *,
    surface: TorusSurface,
    point_data: dict[str, np.ndarray] | None = None,
    cell_data: dict[str, np.ndarray] | None = None,
) -> VTKUnstructuredGrid:
    """Convert a periodic (θ, φ) torus grid into a quad-mesh UnstructuredGrid."""
    xyz = _as_np(surface.r).reshape(-1, 3)
    n_theta = int(surface.theta.size)
    n_phi = int(surface.phi.size)

    i = np.arange(n_theta, dtype=np.int32)[:, None]
    j = np.arange(n_phi, dtype=np.int32)[None, :]
    jp = (j + 1) % n_phi
    ip = (i + 1) % n_theta

    idx00 = i * n_phi + j
    idx10 = ip * n_phi + j
    idx11 = ip * n_phi + jp
    idx01 = i * n_phi + jp
    cells = np.stack([idx00, idx10, idx11, idx01], axis=-1).reshape(-1, 4).astype(np.int32)

    # Default point data (useful for ParaView filters).
    theta = np.asarray(surface.theta, dtype=np.float64)
    phi = np.asarray(surface.phi, dtype=np.float64)
    th_grid = np.repeat(theta, n_phi)
    ph_grid = np.tile(phi, n_theta)

    pd = {"theta": th_grid, "phi": ph_grid}
    if point_data:
        pd.update({k: np.asarray(v) for k, v in point_data.items()})

    return VTKUnstructuredGrid(
        points=xyz,
        cells=cells,
        cell_type=VTK_QUAD,
        point_data=pd,
        cell_data=cell_data,
    )


def point_cloud_to_vtu(
    *,
    points: np.ndarray,
    point_data: dict[str, np.ndarray] | None = None,
) -> VTKUnstructuredGrid:
    """Create a vertex-cell point cloud dataset."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N,3), got {pts.shape}")
    n = int(pts.shape[0])
    cells = np.arange(n, dtype=np.int32)[:, None]
    return VTKUnstructuredGrid(
        points=pts,
        cells=cells,
        cell_type=VTK_VERTEX,
        point_data=point_data or {},
        cell_data=None,
    )


def fieldlines_to_vtu(
    *,
    traj: np.ndarray,
    point_data: dict[str, np.ndarray] | None = None,
    cell_data: dict[str, np.ndarray] | None = None,
) -> VTKUnstructuredGrid:
    """Create a polyline dataset from field line trajectories.

    Parameters
    ----------
    traj:
        Array of shape (n_lines, n_points, 3).
    """
    tr = np.asarray(traj, dtype=np.float64)
    if tr.ndim != 3 or tr.shape[-1] != 3:
        raise ValueError(f"traj must have shape (n_lines,n_points,3), got {tr.shape}")
    n_lines, n_pts_line, _ = tr.shape
    pts = tr.reshape(-1, 3)
    cells = np.arange(n_lines * n_pts_line, dtype=np.int32).reshape(n_lines, n_pts_line)

    pd = point_data or {}
    if "s_index" not in pd:
        pd["s_index"] = np.tile(np.arange(n_pts_line, dtype=np.float64), n_lines)
    if "line_id" not in pd:
        pd["line_id"] = np.repeat(np.arange(n_lines, dtype=np.float64), n_pts_line)

    cd = cell_data or {}
    if "line_id" not in cd:
        cd["line_id"] = np.arange(n_lines, dtype=np.float64)

    return VTKUnstructuredGrid(
        points=pts,
        cells=cells,
        cell_type=VTK_POLY_LINE,
        point_data=pd,
        cell_data=cd,
    )

