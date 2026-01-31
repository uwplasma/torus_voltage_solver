import xml.etree.ElementTree as ET

import numpy as np

from torus_solver import make_torus_surface
from torus_solver.paraview import (
    VTK_POLY_LINE,
    VTK_QUAD,
    VTK_VERTEX,
    fieldlines_to_vtu,
    point_cloud_to_vtu,
    torus_surface_to_vtu,
    write_vtm,
    write_vtu,
)


def _piece(path) -> ET.Element:
    root = ET.parse(path).getroot()
    piece = root.find(".//Piece")
    assert piece is not None, "Missing <Piece> element"
    return piece


def _data_array(piece: ET.Element, *, name: str) -> ET.Element:
    out = None
    for da in piece.findall(".//DataArray"):
        if da.attrib.get("Name") == name:
            out = da
            break
    assert out is not None, f"Missing DataArray Name={name!r}"
    return out


def _read_numbers(da: ET.Element, *, dtype=float) -> np.ndarray:
    txt = da.text or ""
    return np.fromstring(txt, sep=" ", dtype=dtype)


def test_write_vtu_torus_surface_counts_and_arrays(tmp_path):
    surf = make_torus_surface(R0=3.0, a=1.0, n_theta=4, n_phi=5)
    n_points = int(surf.theta.size * surf.phi.size)
    n_cells = n_points  # periodic quad mesh

    V = np.arange(n_points, dtype=float)
    grid = torus_surface_to_vtu(surface=surf, point_data={"V": V})
    path = write_vtu(tmp_path / "surface.vtu", grid)

    piece = _piece(path)
    assert int(piece.attrib["NumberOfPoints"]) == n_points
    assert int(piece.attrib["NumberOfCells"]) == n_cells

    types = _read_numbers(_data_array(piece, name="types"), dtype=np.uint8)
    assert types.shape == (n_cells,)
    assert np.all(types == VTK_QUAD)

    offsets = _read_numbers(_data_array(piece, name="offsets"), dtype=np.int32)
    assert offsets.shape == (n_cells,)
    assert offsets[-1] == 4 * n_cells

    conn = _read_numbers(_data_array(piece, name="connectivity"), dtype=np.int32)
    assert conn.shape == (4 * n_cells,)
    assert int(conn.min()) >= 0
    assert int(conn.max()) < n_points

    V_read = _read_numbers(_data_array(piece, name="V"), dtype=float)
    assert V_read.shape == (n_points,)
    np.testing.assert_allclose(V_read, V)


def test_write_vtu_point_cloud_vertex_cells(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)
    I = np.array([+1.0, -2.0, +0.5], dtype=float)
    grid = point_cloud_to_vtu(points=pts, point_data={"I": I})
    path = write_vtu(tmp_path / "points.vtu", grid)

    piece = _piece(path)
    assert int(piece.attrib["NumberOfPoints"]) == 3
    assert int(piece.attrib["NumberOfCells"]) == 3

    types = _read_numbers(_data_array(piece, name="types"), dtype=np.uint8)
    assert np.all(types == VTK_VERTEX)

    I_read = _read_numbers(_data_array(piece, name="I"), dtype=float)
    np.testing.assert_allclose(I_read, I)


def test_write_vtu_fieldlines_polyline_cells(tmp_path):
    # Two field lines, each with 4 points.
    traj = np.array(
        [
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 0.0], [3.0, 1.0, 0.0]],
        ],
        dtype=float,
    )
    grid = fieldlines_to_vtu(traj=traj)
    path = write_vtu(tmp_path / "lines.vtu", grid)

    piece = _piece(path)
    assert int(piece.attrib["NumberOfPoints"]) == 8
    assert int(piece.attrib["NumberOfCells"]) == 2

    types = _read_numbers(_data_array(piece, name="types"), dtype=np.uint8)
    assert np.all(types == VTK_POLY_LINE)

    offsets = _read_numbers(_data_array(piece, name="offsets"), dtype=np.int32)
    np.testing.assert_allclose(offsets, np.array([4, 8], dtype=np.int32))

    # Default per-point data written by helper.
    line_id = _read_numbers(_data_array(piece, name="line_id"), dtype=float)
    assert set(np.unique(line_id).tolist()) == {0.0, 1.0}


def test_write_vtm_references_files(tmp_path):
    pts = np.array([[0.0, 0.0, 0.0]], dtype=float)
    a = write_vtu(tmp_path / "a.vtu", point_cloud_to_vtu(points=pts))
    b = write_vtu(tmp_path / "b.vtu", point_cloud_to_vtu(points=pts))
    vtm = write_vtm(tmp_path / "scene.vtm", {"a": a, "b": b})

    root = ET.parse(vtm).getroot()
    datasets = root.findall(".//DataSet")
    assert len(datasets) == 2
    files = [d.attrib["file"] for d in datasets]
    assert "a.vtu" in files
    assert "b.vtu" in files
