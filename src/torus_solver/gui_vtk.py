from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

import jax
import jax.numpy as jnp
import optax

from .biot_savart import MU0, biot_savart_surface
from .fieldline import trace_field_lines_batch
from .fields import ideal_toroidal_field
from .optimize import SourceParams
from .poisson import solve_current_potential, surface_current_from_potential
from .sources import deposit_current_sources
from .targets import vmec_target_surface
from .torus import TorusSurface, make_torus_surface
from .vmec import read_vmec_boundary


try:
    import vtk  # type: ignore
    from vtk.util.numpy_support import numpy_to_vtk  # type: ignore
except Exception as e:  # pragma: no cover
    vtk = None
    numpy_to_vtk = None
    _vtk_import_error = e


ScalarName = Literal["|K|", "V", "s", "K_theta", "K_phi"]
CutScalarName = Literal["|K|", "V", "K_theta", "K_phi"]

# Avoid hard runtime dependence on VTK in type hints (important for docs builds).
VtkPolyData = Any
VtkActor = Any
if vtk is not None:  # pragma: no cover
    VtkPolyData = vtk.vtkPolyData
    VtkActor = vtk.vtkActor


@dataclass(frozen=True)
class GUIConfig:
    # Geometry + numerics (keep small for interactivity).
    R0: float = 3.0
    a: float = 1.0
    n_theta: int = 32
    n_phi: int = 32
    sigma_theta: float = 0.25
    sigma_phi: float = 0.25
    sigma_s: float = 1.0
    cg_tol: float = 1e-8
    cg_maxiter: int = 800

    # Electrodes
    n_electrodes_max: int = 32
    current_default_A: float = 1000.0
    current_slider_max_A: float = 3000.0

    # Biot–Savart / tracing
    biot_savart_eps: float = 1e-8
    n_fieldlines: int = 12
    fieldline_steps: int = 500
    fieldline_step_size_m: float = 0.03
    # Optional background field for field-line tracing / visualization:
    # ideal toroidal B ~ 1/R with magnitude Bext0 at R=R0.
    Bext0: float = 1e-4  # Tesla at R=R0 (0 disables)
    bg_field_default_on: bool = False

    # Rendering
    surface_opacity: float = 0.35
    window_size: tuple[int, int] = (1200, 820)


@dataclass(frozen=True)
class CutGUIConfig:
    # Geometry + numerics (keep small for interactivity).
    R0: float = 3.0
    a: float = 0.3
    n_theta: int = 32
    n_phi: int = 32
    theta_cut: float = float(np.pi)  # where the cut/jump appears in the plotted V
    sigma_s: float = 1.0

    # "Battery": voltage drop across the cut (signed).
    V_cut_default: float = 1.0
    V_cut_slider_max: float = 5.0

    # Optional extra electrodes (current sources/sinks) on top of the cut-driven current.
    sigma_theta: float = 0.25
    sigma_phi: float = 0.25
    cg_tol: float = 1e-8
    cg_maxiter: int = 800
    n_electrodes_max: int = 32
    current_default_A: float = 1000.0
    current_slider_max_A: float = 3000.0

    # Biot–Savart / tracing
    biot_savart_eps: float = 1e-8
    n_fieldlines: int = 12
    fieldline_steps: int = 500
    fieldline_step_size_m: float = 0.03
    # Optional background field for field-line tracing / visualization:
    # ideal toroidal B ~ 1/R with magnitude Bext0 at R=R0.
    Bext0: float = 1e-4  # Tesla at R=R0 (0 disables)
    bg_field_default_on: bool = False

    # Rendering
    surface_opacity: float = 0.35
    window_size: tuple[int, int] = (1200, 820)


@dataclass(frozen=True)
class VmecOptGUIConfig:
    # VMEC target surface (boundary) inside the circular torus.
    vmec_input: str = "examples/data/vmec/input.QA_nfp2"
    surf_n_theta: int = 32
    surf_n_phi: int = 56
    fit_margin: float = 0.7

    # Winding surface (circular torus).
    R0: float = 1.0
    a: float = 0.3
    n_theta: int = 32
    n_phi: int = 32

    # Background field (ideal toroidal 1/R).
    B0: float = 1.0  # Tesla at R=R0
    # For diagnostics: allow tracing field lines with/without the background field.
    trace_include_bg_default_on: bool = True

    # Electrodes (fixed size arrays for interactivity).
    n_electrodes_max: int = 64
    n_electrodes_init: int = 32
    current_default_A: float = 1e6
    current_slider_max_A: float = 1e7

    # Electrode model.
    sigma_theta: float = 0.25
    sigma_phi: float = 0.25
    sigma_s: float = 1.0

    # Scaling: I_phys = current_scale * currents_raw (then projected to net-zero over active electrodes).
    current_scale: float | None = None
    init_current_raw_rms: float = 1.0

    # Optimization controls.
    lr: float = 1e-2
    reg_currents: float = 1e-3  # weight on ⟨(I/I_scale)^2⟩
    optimize_positions: bool = True
    steps_per_opt: int = 25

    # Poisson solve.
    cg_tol: float = 1e-10
    cg_maxiter: int = 2000
    use_preconditioner: bool = False

    # Biot–Savart / tracing.
    biot_savart_eps: float = 1e-8
    n_fieldlines: int = 12
    fieldline_steps: int = 500
    fieldline_step_size_m: float = 0.03

    # Rendering.
    surface_opacity: float = 0.35
    target_opacity: float = 0.25
    window_size: tuple[int, int] = (1300, 880)


def _require_vtk() -> None:  # pragma: no cover
    if vtk is None:
        raise ImportError(
            "VTK is required for the interactive GUI. "
            "Install with `pip install vtk` (or `pip install .[gui]` if you add extras)."
        ) from _vtk_import_error


def _resolve_vmec_input_path(vmec_input: str) -> Path:
    p = Path(vmec_input)
    if p.exists():
        return p

    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / vmec_input,  # allow paths relative to repo root
        repo_root / "examples" / vmec_input,
        repo_root / "examples" / "data" / "vmec" / Path(vmec_input).name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    msg = (
        "VMEC input file not found.\n"
        f"  got: {vmec_input}\n"
        f"  tried: {p.resolve()}\n"
        + "".join(f"  tried: {c}\n" for c in candidates)
        + "Tip: run with `--vmec-input examples/data/vmec/input.QA_nfp2` (from the repo root).\n"
    )
    raise FileNotFoundError(msg)


def torus_xyz(R0: float, a: float, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Map (theta,phi) -> xyz on the circular torus."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    R = R0 + a * np.cos(theta)
    x = R * np.cos(phi)
    y = R * np.sin(phi)
    z = a * np.sin(theta)
    return np.stack([x, y, z], axis=-1)


def torus_angles_from_point(R0: float, p: np.ndarray) -> tuple[float, float]:
    """Map xyz -> (theta,phi) for points on/near the torus surface."""
    x, y, z = map(float, p)
    phi = np.arctan2(y, x) % (2 * np.pi)
    R = np.sqrt(x * x + y * y)
    theta = np.arctan2(z, R - R0) % (2 * np.pi)
    return theta, phi


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _cut_phase_theta_np(*, theta: np.ndarray, R0: float, a: float, theta_cut: float) -> np.ndarray:
    """Phase f(θ) in [0,1) with a jump at theta_cut (for plotting a cut potential)."""
    theta = np.asarray(theta, dtype=float)
    theta_cut = float(theta_cut) % (2 * np.pi)

    diff = _wrap_angle_np(theta - theta_cut)
    k0 = int(np.argmin(np.abs(diff)))

    R = R0 + a * np.cos(theta)
    q = 1.0 / R

    q_roll = np.roll(q, -k0)
    dtheta = float(2 * np.pi / theta.size)
    total = float(np.sum(q_roll) * dtheta)
    cum = np.concatenate([[0.0], np.cumsum(q_roll[:-1])]) * dtheta
    f_roll = cum / total
    return np.roll(f_roll, k0)


def _build_torus_polydata(xyz: np.ndarray) -> VtkPolyData:  # pragma: no cover
    _require_vtk()
    n_theta, n_phi, _ = xyz.shape
    pts = xyz.reshape((-1, 3))

    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(numpy_to_vtk(pts, deep=True))
    poly.SetPoints(vtk_points)

    cells = vtk.vtkCellArray()
    for i in range(n_theta):
        i2 = (i + 1) % n_theta
        for j in range(n_phi):
            j2 = (j + 1) % n_phi
            ids = [
                i * n_phi + j,
                i2 * n_phi + j,
                i2 * n_phi + j2,
                i * n_phi + j2,
            ]
            quad = vtk.vtkQuad()
            for k, pid in enumerate(ids):
                quad.GetPointIds().SetId(k, int(pid))
            cells.InsertNextCell(quad)
    poly.SetPolys(cells)
    return poly


def _build_fieldlines_polydata(n_lines: int, n_pts: int) -> VtkPolyData:  # pragma: no cover
    _require_vtk()
    poly = vtk.vtkPolyData()
    points = vtk.vtkPoints()
    points.SetNumberOfPoints(n_lines * n_pts)
    poly.SetPoints(points)

    lines = vtk.vtkCellArray()
    for i in range(n_lines):
        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(n_pts)
        base = i * n_pts
        for k in range(n_pts):
            pl.GetPointIds().SetId(k, base + k)
        lines.InsertNextCell(pl)
    poly.SetLines(lines)
    return poly


class TorusElectrodeGUI:  # pragma: no cover
    """VTK GUI: click to add/move electrodes; slider to change current; see field lines."""

    def __init__(self, cfg: GUIConfig, *, initial_electrodes: dict | None = None):
        _require_vtk()
        jax.config.update("jax_enable_x64", True)

        self.cfg = cfg
        self.surface: TorusSurface = make_torus_surface(
            R0=cfg.R0, a=cfg.a, n_theta=cfg.n_theta, n_phi=cfg.n_phi
        )

        # Electrode state (fixed size to avoid recompilation when user adds/removes).
        self.N = cfg.n_electrodes_max
        self.theta_src = np.zeros((self.N,), dtype=float)
        self.phi_src = np.zeros((self.N,), dtype=float)
        self.currents_raw = np.zeros((self.N,), dtype=float)
        self.active = np.zeros((self.N,), dtype=float)

        if initial_electrodes is not None:
            th = np.asarray(initial_electrodes.get("theta", []), dtype=float)
            ph = np.asarray(initial_electrodes.get("phi", []), dtype=float)
            I = np.asarray(initial_electrodes.get("I", []), dtype=float)
            n0 = min(self.N, th.size, ph.size, I.size)
            self.theta_src[:n0] = th[:n0]
            self.phi_src[:n0] = ph[:n0]
            self.currents_raw[:n0] = I[:n0]
            self.active[:n0] = 1.0

        self.selected: int | None = int(np.argmax(self.active)) if np.any(self.active) else None
        self.mode: Literal["none", "add_source", "add_sink", "move"] = "none"

        # Precompute unit vectors for K decomposition.
        self._e_theta = self.surface.r_theta / self.surface.a
        self._e_phi = self.surface.r_phi / jnp.sqrt(self.surface.G)[..., None]

        # Field line seeds (inside the torus).
        theta_seed = jnp.linspace(0.0, 2 * jnp.pi, cfg.n_fieldlines, endpoint=False)
        rho = 0.5 * cfg.a
        R = cfg.R0 + rho * jnp.cos(theta_seed)
        Z = rho * jnp.sin(theta_seed)
        self._seeds = jnp.stack([R, jnp.zeros_like(R), Z], axis=-1)

        self.scalar_name: ScalarName = "|K|"
        self.show_fieldlines = True
        self.include_bg_field = bool(self.cfg.bg_field_default_on and float(self.cfg.Bext0) != 0.0)

        # Cached computed state (numpy) so we can update visualization (e.g. scalar choice)
        # without re-running the JAX solve.
        self._cache: dict[str, np.ndarray] = {}
        self._traj_cache: np.ndarray | None = None
        self._Iproj_cache: np.ndarray | None = None

        # Text entry ("textbox") state.
        self._edit_mode: Literal["none", "current"] = "none"
        self._edit_buffer: str = ""

        # Build VTK scene.
        self._build_scene()

        # Compile compute function once.
        self._compute_jit = jax.jit(self._compute_state)
        print("Compiling JAX pipeline (first update may take a moment)...")
        self.update_solution()

    def _build_scene(self) -> None:
        cfg = self.cfg

        # Renderer / window / interactor
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(*cfg.window_size)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Torus surface actor (with scalars updated each solve).
        xyz = np.asarray(self.surface.r)
        self.torus_poly = _build_torus_polydata(xyz)

        self.surface_scalars_np = np.zeros((cfg.n_theta * cfg.n_phi,), dtype=np.float32)
        self.surface_scalars_vtk = numpy_to_vtk(self.surface_scalars_np, deep=False)
        self.surface_scalars_vtk.SetName("scalar")
        self.torus_poly.GetPointData().SetScalars(self.surface_scalars_vtk)

        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        self.lut.Build()

        self.torus_mapper = vtk.vtkPolyDataMapper()
        self.torus_mapper.SetInputData(self.torus_poly)
        self.torus_mapper.SetLookupTable(self.lut)
        self.torus_mapper.SetScalarModeToUsePointData()
        self.torus_mapper.ScalarVisibilityOn()

        self.torus_actor = vtk.vtkActor()
        self.torus_actor.SetMapper(self.torus_mapper)
        self.torus_actor.GetProperty().SetOpacity(cfg.surface_opacity)
        self.torus_actor.GetProperty().SetInterpolationToPhong()
        self.torus_actor.GetProperty().SetSpecular(0.2)
        self.torus_actor.GetProperty().SetSpecularPower(30.0)
        self.renderer.AddActor(self.torus_actor)

        # Axis curve for reference.
        axis_phi = np.linspace(0, 2 * np.pi, 200, endpoint=True)
        axis = np.stack([cfg.R0 * np.cos(axis_phi), cfg.R0 * np.sin(axis_phi), 0.0 * axis_phi], axis=-1)
        axis_poly = vtk.vtkPolyData()
        axis_points = vtk.vtkPoints()
        axis_points.SetData(numpy_to_vtk(axis, deep=True))
        axis_poly.SetPoints(axis_points)
        axis_lines = vtk.vtkCellArray()
        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(axis.shape[0])
        for i in range(axis.shape[0]):
            pl.GetPointIds().SetId(i, i)
        axis_lines.InsertNextCell(pl)
        axis_poly.SetLines(axis_lines)
        axis_mapper = vtk.vtkPolyDataMapper()
        axis_mapper.SetInputData(axis_poly)
        axis_actor = vtk.vtkActor()
        axis_actor.SetMapper(axis_mapper)
        axis_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        axis_actor.GetProperty().SetLineWidth(2.0)
        self.renderer.AddActor(axis_actor)

        # Field lines actor (updated each solve).
        n_pts_line = self.cfg.fieldline_steps + 1
        self.field_poly = _build_fieldlines_polydata(self.cfg.n_fieldlines, n_pts_line)
        self.field_points_np = np.zeros(
            (self.cfg.n_fieldlines * n_pts_line, 3), dtype=np.float32
        )
        self.field_points_vtk = numpy_to_vtk(self.field_points_np, deep=False)
        self.field_points_vtk.SetName("field_points")
        self.field_poly.GetPoints().SetData(self.field_points_vtk)
        self.field_mapper = vtk.vtkPolyDataMapper()
        self.field_mapper.SetInputData(self.field_poly)
        self.field_actor = vtk.vtkActor()
        self.field_actor.SetMapper(self.field_mapper)
        self.field_actor.GetProperty().SetColor(0.1, 0.25, 0.9)
        self.field_actor.GetProperty().SetLineWidth(2.0)
        self.renderer.AddActor(self.field_actor)

        # Electrode actors (spheres).
        self.electrode_actors = []
        self._electrode_actor_to_index = {}
        for i in range(self.N):
            src = vtk.vtkSphereSource()
            src.SetThetaResolution(16)
            src.SetPhiResolution(16)
            src.SetRadius(0.05 * self.cfg.a)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetVisibility(False)
            self.renderer.AddActor(actor)

            self.electrode_actors.append((src, actor))
            self._electrode_actor_to_index[actor] = i

        # Text overlay (help + status).
        self.text = vtk.vtkTextActor()
        self.text.GetTextProperty().SetFontSize(16)
        self.text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.text.SetPosition(12, self.cfg.window_size[1] - 190)
        self.renderer.AddActor2D(self.text)

        # Editable numeric "textbox".
        self.input_text = vtk.vtkTextActor()
        self.input_text.GetTextProperty().SetFontSize(16)
        self.input_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.input_text.SetPosition(12, 12)
        self.renderer.AddActor2D(self.input_text)

        # Current slider for the selected electrode.
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(-self.cfg.current_slider_max_A)
        rep.SetMaximumValue(+self.cfg.current_slider_max_A)
        rep.SetValue(0.0)
        rep.SetTitleText("Selected electrode current I [A]")
        rep.SetLabelFormat("%0.0f")
        rep.SetSliderLength(0.02)
        rep.SetSliderWidth(0.03)
        rep.SetTubeWidth(0.006)
        rep.SetEndCapLength(0.01)
        rep.SetEndCapWidth(0.03)
        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(0.10, 0.06)
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetValue(0.55, 0.06)

        self.slider_rep = rep
        self.slider = vtk.vtkSliderWidget()
        self.slider.SetInteractor(self.interactor)
        self.slider.SetRepresentation(rep)
        self.slider.EnabledOn()

        def on_slider_end(_obj, _evt):
            if self.selected is None:
                return
            val = float(self.slider_rep.GetValue())
            self.currents_raw[self.selected] = val
            self.update_solution()

        self.slider.AddObserver(vtk.vtkCommand.EndInteractionEvent, on_slider_end)

        # Picking helpers.
        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.0005)
        self.prop_picker = vtk.vtkPropPicker()

        # Interactor events.
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self._on_keypress)
        self.interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self._on_left_click)

        # Initial camera.
        self.renderer.ResetCamera()

    def _help_text(self) -> str:
        return (
            "Torus electrode GUI (VTK)\n"
            "Mouse: rotate/zoom as usual\n"
            "Click electrode: select\n"
            "Keys:\n"
            "  a: add SOURCE (next click on surface)\n"
            "  z: add SINK   (next click on surface)\n"
            "  m: move selected (next click on surface)\n"
            "  d: delete selected\n"
            "  tab: cycle selected\n"
            "  c: cycle surface scalar (|K|, V, s, Kθ, Kφ)\n"
            "  f: toggle field lines\n"
            "  b: toggle external toroidal field (ideal 1/R)\n"
            "  r: recompute\n"
            "  e: export ParaView (.vtu/.vtm)\n"
            "  i (or v): type selected I\n"
            "  s: save screenshot\n"
        )

    def _status_text(self) -> str:
        n_active = int(np.sum(self.active))
        sel = self.selected
        if sel is None:
            sel_txt = "none"
        else:
            Iraw = float(self.currents_raw[sel])
            Iproj = None
            if self._Iproj_cache is not None and sel < self._Iproj_cache.size:
                Iproj = float(self._Iproj_cache[sel])
            if Iproj is None:
                sel_txt = f"{sel}  Iraw={Iraw:+.0f} A  (active={self.active[sel]:.0f})"
            else:
                sel_txt = (
                    f"{sel}  Iraw={Iraw:+.0f} A  Iproj={Iproj:+.0f} A  (active={self.active[sel]:.0f})"
                )
        return (
            f"Active electrodes: {n_active}/{self.N}\n"
            f"Selected: {sel_txt}\n"
            f"Mode: {self.mode}\n"
            f"Scalar: {self.scalar_name}\n"
            f"external Bphi~1/R: {'ON' if self.include_bg_field else 'OFF'}  "
            f"(Bext0={float(self.cfg.Bext0):.3g} T at R0={self.cfg.R0:g} m)\n"
            f"sigma_theta={self.cfg.sigma_theta:.3f}  sigma_phi={self.cfg.sigma_phi:.3f}\n"
            f"fieldlines: {self.cfg.n_fieldlines}  steps: {self.cfg.fieldline_steps}  ds: {self.cfg.fieldline_step_size_m}\n"
        )

    def _update_text(self) -> None:
        self.text.SetInput(self._help_text() + "\n" + self._status_text())
        if self._edit_mode == "none":
            self.input_text.SetInput("Type: press 'i' (or 'v') to enter the selected electrode current I.")
        else:
            self.input_text.SetInput(
                f"Input [current]: {self._edit_buffer}   (Enter=apply, Esc=cancel)"
            )

    def _project_currents(self) -> np.ndarray:
        mask = self.active.astype(float)
        I = self.currents_raw * mask
        n = float(np.sum(mask))
        if n <= 0:
            return np.zeros_like(I)
        mean = float(np.sum(I) / n)
        return I - mean * mask

    def _apply_surface_scalar(self) -> None:
        if not self._cache:
            return

        if self.scalar_name == "|K|":
            scal = self._cache["Kmag"].reshape((-1,))
        elif self.scalar_name == "V":
            scal = self._cache["V"].reshape((-1,))
        elif self.scalar_name == "s":
            scal = self._cache["s"].reshape((-1,))
        elif self.scalar_name == "K_theta":
            scal = self._cache["Ktheta"].reshape((-1,))
        elif self.scalar_name == "K_phi":
            scal = self._cache["Kphi"].reshape((-1,))
        else:
            raise ValueError(self.scalar_name)

        # Update VTK scalar array in-place.
        self.surface_scalars_np[:] = scal.astype(np.float32, copy=False)
        self.surface_scalars_vtk.Modified()
        self.torus_poly.Modified()

        # Adjust color range robustly.
        smin = float(np.nanmin(scal))
        smax = float(np.nanmax(scal))
        if not np.isfinite(smin) or not np.isfinite(smax) or smin == smax:
            smin, smax = 0.0, 1.0
        if self.scalar_name in ("s", "K_theta", "K_phi", "V"):
            vmax = max(abs(smin), abs(smax))
            smin, smax = -vmax, vmax
        self.torus_mapper.SetScalarRange(smin, smax)

    def _update_electrode_actors(self) -> None:
        Iproj = self._Iproj_cache
        if Iproj is None:
            Iproj = self._project_currents()

        for i in range(self.N):
            src, actor = self.electrode_actors[i]
            if self.active[i] <= 0.0:
                actor.SetVisibility(False)
                continue
            actor.SetVisibility(True)

            p = torus_xyz(self.cfg.R0, self.cfg.a, self.theta_src[i], self.phi_src[i])
            src.SetCenter(float(p[0]), float(p[1]), float(p[2]))

            I = float(Iproj[i])
            if i == self.selected:
                actor.GetProperty().SetColor(1.0, 0.8, 0.2)
            else:
                if I > 0:
                    actor.GetProperty().SetColor(0.85, 0.15, 0.15)
                elif I < 0:
                    actor.GetProperty().SetColor(0.15, 0.25, 0.85)
                else:
                    actor.GetProperty().SetColor(0.4, 0.4, 0.4)

            r0 = 0.04 * self.cfg.a
            r = r0 * (0.6 + 0.8 * min(abs(I) / self.cfg.current_slider_max_A, 1.0))
            src.SetRadius(float(r))
            src.Modified()

    def _compute_state(
        self,
        theta_src: jnp.ndarray,
        phi_src: jnp.ndarray,
        currents_raw: jnp.ndarray,
        active: jnp.ndarray,
        include_bg_field: jnp.ndarray,
        compute_traj: jnp.ndarray,
    ):
        # Project currents to net-zero over active electrodes.
        mask = active
        I = currents_raw * mask
        n = jnp.sum(mask)
        mean = jnp.where(n > 0, jnp.sum(I) / n, 0.0)
        I = I - mean * mask

        s = deposit_current_sources(
            self.surface,
            theta_src=theta_src,
            phi_src=phi_src,
            currents=I,
            sigma_theta=self.cfg.sigma_theta,
            sigma_phi=self.cfg.sigma_phi,
        )
        V, _ = solve_current_potential(
            self.surface, s, tol=self.cfg.cg_tol, maxiter=self.cfg.cg_maxiter
        )
        K = surface_current_from_potential(self.surface, V, sigma_s=self.cfg.sigma_s)

        Kmag = jnp.linalg.norm(K, axis=-1)
        Ktheta = jnp.sum(K * self._e_theta, axis=-1)
        Kphi = jnp.sum(K * self._e_phi, axis=-1)

        def B_fn(xyz: jnp.ndarray) -> jnp.ndarray:
            B = biot_savart_surface(self.surface, K, xyz, eps=self.cfg.biot_savart_eps)
            if float(self.cfg.Bext0) != 0.0:
                B = B + jnp.asarray(include_bg_field, dtype=B.dtype) * ideal_toroidal_field(
                    xyz, B0=float(self.cfg.Bext0), R0=float(self.cfg.R0)
                )
            return B

        n_steps = self.cfg.fieldline_steps
        n_lines = self.cfg.n_fieldlines

        def do_trace(_):
            return trace_field_lines_batch(
                B_fn,
                self._seeds,
                step_size=self.cfg.fieldline_step_size_m,
                n_steps=n_steps,
                normalize=True,
            )

        def no_trace(_):
            return jnp.zeros((n_steps + 1, n_lines, 3), dtype=jnp.float64)

        traj = jax.lax.cond(compute_traj, do_trace, no_trace, operand=None)

        return V, s, Kmag, Ktheta, Kphi, traj, I

    def update_solution(self) -> None:
        t0 = time.perf_counter()
        self._update_text()
        self.window.Render()

        th = jnp.asarray(self.theta_src)
        ph = jnp.asarray(self.phi_src)
        Iraw = jnp.asarray(self.currents_raw)
        act = jnp.asarray(self.active)

        V, s, Kmag, Ktheta, Kphi, traj, Iproj = self._compute_jit(
            th,
            ph,
            Iraw,
            act,
            jnp.asarray(self.include_bg_field),
            jnp.asarray(self.show_fieldlines),
        )
        V.block_until_ready()
        t1 = time.perf_counter()

        # Cache computed state for fast GUI updates (cycle scalars, toggle lines, selection).
        self._cache = {
            "V": np.asarray(V, dtype=np.float32),
            "s": np.asarray(s, dtype=np.float32),
            "Kmag": np.asarray(Kmag, dtype=np.float32),
            "Ktheta": np.asarray(Ktheta, dtype=np.float32),
            "Kphi": np.asarray(Kphi, dtype=np.float32),
        }
        self._traj_cache = np.asarray(traj, dtype=np.float32)
        self._Iproj_cache = np.asarray(Iproj, dtype=float)

        self._apply_surface_scalar()
        self._update_electrode_actors()

        # Update field lines geometry.
        self.field_actor.SetVisibility(bool(self.show_fieldlines))
        if self.show_fieldlines and self._traj_cache is not None:
            self.field_points_np[:] = self._traj_cache.reshape((-1, 3))
            self.field_points_vtk.Modified()
            self.field_poly.Modified()

        if self.selected is not None:
            self.slider_rep.SetValue(float(self.currents_raw[self.selected]))
        self._update_text()

        self.window.Render()
        t2 = time.perf_counter()
        print(
            "update: solve+trace {:.3f}s, total {:.3f}s  (scalar={}, active={})".format(
                t1 - t0, t2 - t0, self.scalar_name, int(np.sum(self.active))
            )
        )

    def _select_next(self) -> None:
        active_idx = np.flatnonzero(self.active > 0.0)
        if active_idx.size == 0:
            self.selected = None
            return
        if self.selected is None or self.selected not in active_idx:
            self.selected = int(active_idx[0])
            return
        k = int(np.where(active_idx == self.selected)[0][0])
        self.selected = int(active_idx[(k + 1) % active_idx.size])

    def _delete_selected(self) -> None:
        if self.selected is None:
            return
        i = self.selected
        self.active[i] = 0.0
        self.currents_raw[i] = 0.0
        self.theta_src[i] = 0.0
        self.phi_src[i] = 0.0
        self._select_next()
        self.update_solution()

    def _add_electrode(self, theta: float, phi: float, current: float) -> None:
        free = np.flatnonzero(self.active <= 0.0)
        if free.size == 0:
            print("No free electrode slots; increase n_electrodes_max.")
            return
        i = int(free[0])
        self.theta_src[i] = float(theta)
        self.phi_src[i] = float(phi)
        self.currents_raw[i] = float(current)
        self.active[i] = 1.0
        self.selected = i
        self.slider_rep.SetValue(float(self.currents_raw[i]))
        self.update_solution()

    def _on_keypress(self, _obj, _evt) -> None:
        key_sym = self.interactor.GetKeySym()
        key_code = self.interactor.GetKeyCode()

        if self._edit_mode != "none":
            if key_sym in ("Escape",):
                self._edit_mode = "none"
                self._edit_buffer = ""
                self._update_text()
                self.window.Render()
                return
            if key_sym in ("Return", "KP_Enter"):
                try:
                    val = float(self._edit_buffer.strip())
                except Exception:
                    print(f"Could not parse number: {self._edit_buffer!r}")
                    return
                if self.selected is not None:
                    self.currents_raw[self.selected] = float(val)
                    self.slider_rep.SetValue(float(self.currents_raw[self.selected]))
                    self._edit_mode = "none"
                    self._edit_buffer = ""
                    self.update_solution()
                return
            if key_sym in ("BackSpace", "Delete"):
                self._edit_buffer = self._edit_buffer[:-1]
                self._update_text()
                self.window.Render()
                return
            if key_code and key_code in "0123456789+-eE.":
                self._edit_buffer = self._edit_buffer + key_code
                self._update_text()
                self.window.Render()
                return
            return

        if key_sym in ("i", "I", "v", "V"):
            if self.selected is None:
                return
            self._edit_mode = "current"
            self._edit_buffer = f"{float(self.currents_raw[self.selected]):.6g}"
            self._update_text()
            self.window.Render()
            return

        if key_sym in ("a", "A"):
            self.mode = "add_source"
        elif key_sym in ("z", "Z"):
            self.mode = "add_sink"
        elif key_sym in ("m", "M"):
            self.mode = "move"
        elif key_sym in ("d", "Delete", "BackSpace"):
            self._delete_selected()
            return
        elif key_sym in ("Tab",):
            self._select_next()
            self._update_electrode_actors()
        elif key_sym in ("c", "C"):
            order: list[ScalarName] = ["|K|", "V", "s", "K_theta", "K_phi"]
            k = order.index(self.scalar_name)
            self.scalar_name = order[(k + 1) % len(order)]
            self._apply_surface_scalar()
        elif key_sym in ("f", "F"):
            self.show_fieldlines = not self.show_fieldlines
            if self.show_fieldlines:
                self.update_solution()
                return
            self.field_actor.SetVisibility(False)
        elif key_sym in ("b", "B"):
            self.include_bg_field = not self.include_bg_field
            print(
                f"External toroidal field (1/R): {'ON' if self.include_bg_field else 'OFF'} "
                f"(Bext0={float(self.cfg.Bext0):.3g} T at R0={self.cfg.R0:g} m)"
            )
            if self.show_fieldlines:
                self.update_solution()
                return
        elif key_sym in ("r", "R"):
            self.update_solution()
            return
        elif key_sym in ("e", "E"):
            self._export_paraview()
        elif key_sym in ("s", "S"):
            self._save_screenshot()
        self._update_text()
        self.window.Render()

    def _on_left_click(self, _obj, _evt) -> None:
        x, y = self.interactor.GetEventPosition()

        # 1) Try selecting an electrode.
        self.prop_picker.Pick(x, y, 0, self.renderer)
        actor = self.prop_picker.GetActor()
        if actor in self._electrode_actor_to_index:
            self.selected = int(self._electrode_actor_to_index[actor])
            self.slider_rep.SetValue(float(self.currents_raw[self.selected]))
            self._update_electrode_actors()
            self._update_text()
            self.window.Render()
            return

        # 2) Add/move electrode by picking torus surface.
        if self.mode in ("add_source", "add_sink", "move"):
            if not self.cell_picker.Pick(x, y, 0, self.renderer):
                self.mode = "none"
                self._update_text()
                self.window.Render()
                return
            p = np.array(self.cell_picker.GetPickPosition(), dtype=float)
            theta, phi = torus_angles_from_point(self.cfg.R0, p)

            if self.mode == "move":
                if self.selected is not None and self.active[self.selected] > 0:
                    self.theta_src[self.selected] = theta
                    self.phi_src[self.selected] = phi
                    self.mode = "none"
                    self.update_solution()
                else:
                    self.mode = "none"
            elif self.mode == "add_source":
                self.mode = "none"
                self._add_electrode(theta, phi, +self.cfg.current_default_A)
            elif self.mode == "add_sink":
                self.mode = "none"
                self._add_electrode(theta, phi, -self.cfg.current_default_A)
            return

        # Fall back to default camera interaction.
        self.interactor.GetInteractorStyle().OnLeftButtonDown()

    def _save_screenshot(self) -> None:
        outdir = Path("figures/gui_screenshots")
        outdir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = outdir / f"torus_gui_{ts}.png"

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.window)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"Saved screenshot: {path}")

    def _export_paraview(self) -> None:
        from .paraview import fieldlines_to_vtu, point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu

        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = Path("paraview") / f"gui_torus_electrodes_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)

        V = self._cache.get("V")
        s = self._cache.get("s")
        Ktheta = self._cache.get("Ktheta")
        Kphi = self._cache.get("Kphi")
        Kmag = self._cache.get("Kmag")
        if V is None or s is None or Ktheta is None or Kphi is None or Kmag is None:
            print("ParaView export: no cached solution yet.")
            return

        e_theta = np.asarray(self._e_theta, dtype=float)
        e_phi = np.asarray(self._e_phi, dtype=float)
        K_vec = Ktheta[..., None] * e_theta + Kphi[..., None] * e_phi

        surf = write_vtu(
            outdir / "winding_surface.vtu",
            torus_surface_to_vtu(
                surface=self.surface,
                point_data={
                    "V": V.reshape(-1),
                    "s": s.reshape(-1),
                    "K": K_vec.reshape(-1, 3),
                    "Ktheta": Ktheta.reshape(-1),
                    "Kphi": Kphi.reshape(-1),
                    "|K|": Kmag.reshape(-1),
                },
            ),
        )

        blocks: dict[str, str] = {"winding_surface": surf.name}

        active = np.flatnonzero(self.active > 0.0)
        if active.size > 0:
            xyz = torus_xyz(self.cfg.R0, self.cfg.a, self.theta_src[active], self.phi_src[active])
            I = (
                np.asarray(self._Iproj_cache, dtype=float)[active]
                if self._Iproj_cache is not None
                else np.asarray(self.currents_raw, dtype=float)[active]
            )
            elec = write_vtu(
                outdir / "electrodes.vtu",
                point_cloud_to_vtu(
                    points=np.asarray(xyz, dtype=float),
                    point_data={"I_A": I, "sign_I": np.sign(I)},
                ),
            )
            blocks["electrodes"] = elec.name

        if self.show_fieldlines and self._traj_cache is not None:
            traj_pv = np.transpose(self._traj_cache, (1, 0, 2))
            fl = write_vtu(outdir / "fieldlines.vtu", fieldlines_to_vtu(traj=traj_pv))
            blocks["fieldlines"] = fl.name

        scene = write_vtm(outdir / "scene.vtm", blocks)
        print(f"Saved ParaView scene: {scene}")

    def run(self) -> None:
        print("Starting GUI. Close the window to exit.")
        self._update_text()
        self.window.Render()
        self.interactor.Initialize()
        self.interactor.Start()


def run_torus_electrode_gui(
    *,
    cfg: GUIConfig = GUIConfig(),
    initial_electrodes: dict | None = None,
) -> None:  # pragma: no cover
    """Entry point for examples."""
    _require_vtk()
    print("Interactive torus electrode GUI")
    print(f"  R0={cfg.R0} a={cfg.a} n_theta={cfg.n_theta} n_phi={cfg.n_phi}")
    print(
        f"  sigma_theta={cfg.sigma_theta} sigma_phi={cfg.sigma_phi}  "
        f"n_lines={cfg.n_fieldlines} steps={cfg.fieldline_steps} ds={cfg.fieldline_step_size_m}"
    )
    print(f"  mu0={float(MU0):.6e}")
    app = TorusElectrodeGUI(cfg, initial_electrodes=initial_electrodes)
    app.run()


class TorusCutVoltageGUI:  # pragma: no cover
    """VTK GUI: a toroidal cut voltage drives poloidal current; optional extra electrodes add sources/sinks."""

    def __init__(self, cfg: CutGUIConfig, *, initial_electrodes: dict | None = None):
        _require_vtk()
        jax.config.update("jax_enable_x64", True)

        self.cfg = cfg
        self.surface: TorusSurface = make_torus_surface(
            R0=cfg.R0, a=cfg.a, n_theta=cfg.n_theta, n_phi=cfg.n_phi
        )

        # Axisymmetric cut-driven solution uses ∂θV = C/R, with C fixed by V_cut.
        Rtheta = self.surface.R[:, 0]  # (Nθ,)
        self._I1 = jnp.sum((1.0 / Rtheta) * self.surface.dtheta)  # ∮ dθ/R(θ)

        f = _cut_phase_theta_np(
            theta=np.asarray(self.surface.theta),
            R0=cfg.R0,
            a=cfg.a,
            theta_cut=cfg.theta_cut,
        )
        self._f_theta = jnp.asarray(f, dtype=jnp.float64)  # (Nθ,)

        # Electrode state (fixed size to avoid recompilation when user adds/removes).
        self.N = cfg.n_electrodes_max
        self.theta_src = np.zeros((self.N,), dtype=float)
        self.phi_src = np.zeros((self.N,), dtype=float)
        self.currents_raw = np.zeros((self.N,), dtype=float)
        self.active = np.zeros((self.N,), dtype=float)

        if initial_electrodes is not None:
            th = np.asarray(initial_electrodes.get("theta", []), dtype=float)
            ph = np.asarray(initial_electrodes.get("phi", []), dtype=float)
            I = np.asarray(initial_electrodes.get("I", []), dtype=float)
            n0 = min(self.N, th.size, ph.size, I.size)
            self.theta_src[:n0] = th[:n0]
            self.phi_src[:n0] = ph[:n0]
            self.currents_raw[:n0] = I[:n0]
            self.active[:n0] = 1.0

        self.selected: int | None = int(np.argmax(self.active)) if np.any(self.active) else None
        self.mode: Literal["none", "add_source", "add_sink", "move"] = "none"

        # Precompute unit vectors for K decomposition.
        self._e_theta = self.surface.r_theta / self.surface.a
        self._e_phi = self.surface.r_phi / jnp.sqrt(self.surface.G)[..., None]

        # Field line seeds (inside the torus).
        theta_seed = jnp.linspace(0.0, 2 * jnp.pi, cfg.n_fieldlines, endpoint=False)
        rho = 0.5 * cfg.a
        R = cfg.R0 + rho * jnp.cos(theta_seed)
        Z = rho * jnp.sin(theta_seed)
        self._seeds = jnp.stack([R, jnp.zeros_like(R), Z], axis=-1)

        self.scalar_name: ScalarName = "|K|"
        self.show_fieldlines = True
        self.include_bg_field = bool(self.cfg.bg_field_default_on and float(self.cfg.Bext0) != 0.0)
        self.V_cut = float(cfg.V_cut_default)

        # Cached computed state (numpy) so we can update visualization without recomputing.
        self._cache: dict[str, np.ndarray] = {}
        self._traj_cache: np.ndarray | None = None
        self._Iproj_cache: np.ndarray | None = None

        # Text entry ("textbox") state.
        self._edit_mode: Literal["none", "V_cut", "current"] = "none"
        self._edit_buffer: str = ""

        self._build_scene()

        self._compute_jit = jax.jit(self._compute_state)
        print("Compiling JAX pipeline (first update may take a moment)...")
        self.update_solution()

    def _build_scene(self) -> None:
        cfg = self.cfg

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(*cfg.window_size)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        xyz = np.asarray(self.surface.r)
        self.torus_poly = _build_torus_polydata(xyz)

        self.surface_scalars_np = np.zeros((cfg.n_theta * cfg.n_phi,), dtype=np.float32)
        self.surface_scalars_vtk = numpy_to_vtk(self.surface_scalars_np, deep=False)
        self.surface_scalars_vtk.SetName("scalar")
        self.torus_poly.GetPointData().SetScalars(self.surface_scalars_vtk)

        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        self.lut.Build()

        self.torus_mapper = vtk.vtkPolyDataMapper()
        self.torus_mapper.SetInputData(self.torus_poly)
        self.torus_mapper.SetLookupTable(self.lut)
        self.torus_mapper.SetScalarModeToUsePointData()
        self.torus_mapper.ScalarVisibilityOn()

        self.torus_actor = vtk.vtkActor()
        self.torus_actor.SetMapper(self.torus_mapper)
        self.torus_actor.GetProperty().SetOpacity(cfg.surface_opacity)
        self.torus_actor.GetProperty().SetInterpolationToPhong()
        self.torus_actor.GetProperty().SetSpecular(0.2)
        self.torus_actor.GetProperty().SetSpecularPower(30.0)
        self.renderer.AddActor(self.torus_actor)

        # Axis curve for reference.
        axis_phi = np.linspace(0, 2 * np.pi, 200, endpoint=True)
        axis = np.stack([cfg.R0 * np.cos(axis_phi), cfg.R0 * np.sin(axis_phi), 0.0 * axis_phi], axis=-1)
        axis_poly = vtk.vtkPolyData()
        axis_points = vtk.vtkPoints()
        axis_points.SetData(numpy_to_vtk(axis, deep=True))
        axis_poly.SetPoints(axis_points)
        axis_lines = vtk.vtkCellArray()
        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(axis.shape[0])
        for i in range(axis.shape[0]):
            pl.GetPointIds().SetId(i, i)
        axis_lines.InsertNextCell(pl)
        axis_poly.SetLines(axis_lines)
        axis_mapper = vtk.vtkPolyDataMapper()
        axis_mapper.SetInputData(axis_poly)
        axis_actor = vtk.vtkActor()
        axis_actor.SetMapper(axis_mapper)
        axis_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        axis_actor.GetProperty().SetLineWidth(2.0)
        self.renderer.AddActor(axis_actor)

        # Cut curve + terminals for reference (where V jumps in the visualization).
        cut_phi = np.linspace(0.0, 2 * np.pi, 240, endpoint=True)
        cut_theta = float(cfg.theta_cut) % (2 * np.pi)
        cut = torus_xyz(cfg.R0, cfg.a, cut_theta * np.ones_like(cut_phi), cut_phi)

        def _polyline_actor(points_xyz: np.ndarray, *, rgb: tuple[float, float, float], width: float) -> VtkActor:
            poly = vtk.vtkPolyData()
            pts = vtk.vtkPoints()
            pts.SetData(numpy_to_vtk(points_xyz, deep=True))
            poly.SetPoints(pts)
            lines = vtk.vtkCellArray()
            pl = vtk.vtkPolyLine()
            pl.GetPointIds().SetNumberOfIds(points_xyz.shape[0])
            for i in range(points_xyz.shape[0]):
                pl.GetPointIds().SetId(i, i)
            lines.InsertNextCell(pl)
            poly.SetLines(lines)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(poly)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor(*rgb)
            actor.GetProperty().SetLineWidth(width)
            return actor

        self.cut_actor = _polyline_actor(cut, rgb=(0.15, 0.15, 0.15), width=3.0)
        self.renderer.AddActor(self.cut_actor)

        # Two nearby rings show the "before" and "after" sides of the cut.
        delta = 0.5 * float(self.surface.dtheta)
        th_before = (cut_theta - delta) % (2 * np.pi)
        th_after = (cut_theta + 0.0) % (2 * np.pi)
        ring_before = torus_xyz(cfg.R0, cfg.a, th_before * np.ones_like(cut_phi), cut_phi)
        ring_after = torus_xyz(cfg.R0, cfg.a, th_after * np.ones_like(cut_phi), cut_phi)
        self.cut_before_actor = _polyline_actor(ring_before, rgb=(0.85, 0.15, 0.15), width=2.5)
        self.cut_after_actor = _polyline_actor(ring_after, rgb=(0.15, 0.25, 0.85), width=2.5)
        self.renderer.AddActor(self.cut_before_actor)
        self.renderer.AddActor(self.cut_after_actor)

        # Field lines actor (updated each solve).
        n_pts_line = self.cfg.fieldline_steps + 1
        self.field_poly = _build_fieldlines_polydata(self.cfg.n_fieldlines, n_pts_line)
        self.field_points_np = np.zeros((self.cfg.n_fieldlines * n_pts_line, 3), dtype=np.float32)
        self.field_points_vtk = numpy_to_vtk(self.field_points_np, deep=False)
        self.field_points_vtk.SetName("field_points")
        self.field_poly.GetPoints().SetData(self.field_points_vtk)

        self.field_mapper = vtk.vtkPolyDataMapper()
        self.field_mapper.SetInputData(self.field_poly)
        self.field_actor = vtk.vtkActor()
        self.field_actor.SetMapper(self.field_mapper)
        self.field_actor.GetProperty().SetColor(0.1, 0.25, 0.9)
        self.field_actor.GetProperty().SetLineWidth(2.0)
        self.renderer.AddActor(self.field_actor)

        # Electrode actors (spheres).
        self.electrode_actors = []
        self._electrode_actor_to_index = {}
        for i in range(self.N):
            src = vtk.vtkSphereSource()
            src.SetThetaResolution(16)
            src.SetPhiResolution(16)
            src.SetRadius(0.05 * self.cfg.a)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetVisibility(False)
            self.renderer.AddActor(actor)

            self.electrode_actors.append((src, actor))
            self._electrode_actor_to_index[actor] = i

        # Text overlays: help+status (top) and an editable numeric "textbox" (bottom).
        self.text = vtk.vtkTextActor()
        self.text.GetTextProperty().SetFontSize(16)
        self.text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.text.SetPosition(12, self.cfg.window_size[1] - 210)
        self.renderer.AddActor2D(self.text)

        self.input_text = vtk.vtkTextActor()
        self.input_text.GetTextProperty().SetFontSize(16)
        self.input_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.input_text.SetPosition(12, 12)
        self.renderer.AddActor2D(self.input_text)

        # V_cut slider.
        rep_v = vtk.vtkSliderRepresentation2D()
        rep_v.SetMinimumValue(-self.cfg.V_cut_slider_max)
        rep_v.SetMaximumValue(+self.cfg.V_cut_slider_max)
        rep_v.SetValue(self.V_cut)
        rep_v.SetTitleText("Cut voltage V_cut [arb]  (press 'v' to type)")
        rep_v.SetLabelFormat("%0.3f")
        rep_v.SetSliderLength(0.02)
        rep_v.SetSliderWidth(0.03)
        rep_v.SetTubeWidth(0.006)
        rep_v.SetEndCapLength(0.01)
        rep_v.SetEndCapWidth(0.03)
        rep_v.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep_v.GetPoint1Coordinate().SetValue(0.10, 0.06)
        rep_v.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep_v.GetPoint2Coordinate().SetValue(0.55, 0.06)

        self.V_slider_rep = rep_v
        self.V_slider = vtk.vtkSliderWidget()
        self.V_slider.SetInteractor(self.interactor)
        self.V_slider.SetRepresentation(rep_v)
        self.V_slider.EnabledOn()

        def on_v_slider_end(_obj, _evt):
            self.V_cut = float(self.V_slider_rep.GetValue())
            self.update_solution()

        self.V_slider.AddObserver(vtk.vtkCommand.EndInteractionEvent, on_v_slider_end)

        # Current slider for the selected electrode.
        rep_i = vtk.vtkSliderRepresentation2D()
        rep_i.SetMinimumValue(-self.cfg.current_slider_max_A)
        rep_i.SetMaximumValue(+self.cfg.current_slider_max_A)
        rep_i.SetValue(0.0)
        rep_i.SetTitleText("Selected electrode current I [A]  (press 'i' to type)")
        rep_i.SetLabelFormat("%0.0f")
        rep_i.SetSliderLength(0.02)
        rep_i.SetSliderWidth(0.03)
        rep_i.SetTubeWidth(0.006)
        rep_i.SetEndCapLength(0.01)
        rep_i.SetEndCapWidth(0.03)
        rep_i.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep_i.GetPoint1Coordinate().SetValue(0.58, 0.06)
        rep_i.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep_i.GetPoint2Coordinate().SetValue(0.98, 0.06)

        self.I_slider_rep = rep_i
        self.I_slider = vtk.vtkSliderWidget()
        self.I_slider.SetInteractor(self.interactor)
        self.I_slider.SetRepresentation(rep_i)
        self.I_slider.EnabledOn()

        def on_i_slider_end(_obj, _evt):
            if self.selected is None:
                return
            val = float(self.I_slider_rep.GetValue())
            self.currents_raw[self.selected] = val
            self.update_solution()

        self.I_slider.AddObserver(vtk.vtkCommand.EndInteractionEvent, on_i_slider_end)

        # Picking helpers.
        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.0005)
        self.prop_picker = vtk.vtkPropPicker()

        # Interactor events.
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self._on_keypress)
        self.interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self._on_left_click)

        self.renderer.ResetCamera()

    def _help_text(self) -> str:
        return (
            "Torus cut+electrodes GUI (VTK)\n"
            "Mouse: rotate/zoom as usual\n"
            "Electrodes: red=source (+I), blue=sink (-I), yellow=selected\n"
            "Cut terminals: red ring = higher V side, blue ring = lower V side; thick gray ring = cut location\n"
            "Keys:\n"
            "  a/z/m: add source / add sink / move selected (next click)\n"
            "  d: delete selected    tab: cycle selected\n"
            "  c: cycle scalar (|K|, V, s, Kθ, Kφ)\n"
            "  f: toggle field lines    r: recompute\n"
            "  b: toggle external toroidal field (ideal 1/R)\n"
            "  v: type V_cut    i: type selected I\n"
            "  e: export ParaView (.vtu/.vtm)\n"
            "  s: save screenshot\n"
        )

    def _status_text(self) -> str:
        n_active = int(np.sum(self.active))
        if self.selected is None:
            sel_txt = "none"
        else:
            Iraw = float(self.currents_raw[self.selected])
            Iproj = None
            if self._Iproj_cache is not None and self.selected < self._Iproj_cache.size:
                Iproj = float(self._Iproj_cache[self.selected])
            if Iproj is None:
                sel_txt = f"{self.selected}  Iraw={Iraw:+.0f} A"
            else:
                sel_txt = f"{self.selected}  Iraw={Iraw:+.0f} A  Iproj={Iproj:+.0f} A"

        return (
            f"V_cut={self.V_cut:+.3f}  theta_cut={float(self.cfg.theta_cut)%(2*np.pi):.3f}  sigma_s={self.cfg.sigma_s:.3f}\n"
            f"Active electrodes: {n_active}/{self.N}   Selected: {sel_txt}   Mode: {self.mode}\n"
            f"Scalar: {self.scalar_name}\n"
            f"external Bphi~1/R: {'ON' if self.include_bg_field else 'OFF'}  "
            f"(Bext0={float(self.cfg.Bext0):.3g} T at R0={self.cfg.R0:g} m)\n"
            f"sigma_theta={self.cfg.sigma_theta:.3f}  sigma_phi={self.cfg.sigma_phi:.3f}\n"
            f"fieldlines: {self.cfg.n_fieldlines}  steps: {self.cfg.fieldline_steps}  ds: {self.cfg.fieldline_step_size_m}\n"
        )

    def _update_text(self) -> None:
        self.text.SetInput(self._help_text() + "\n" + self._status_text())

        if self._edit_mode == "none":
            self.input_text.SetInput("Type: press 'v' (V_cut) or 'i' (selected I) to enter a value.")
        else:
            self.input_text.SetInput(
                f"Input [{self._edit_mode}]: {self._edit_buffer}   (Enter=apply, Esc=cancel)"
            )

    def _project_currents(self) -> np.ndarray:
        mask = self.active.astype(float)
        I = self.currents_raw * mask
        n = float(np.sum(mask))
        if n <= 0:
            return np.zeros_like(I)
        mean = float(np.sum(I) / n)
        return I - mean * mask

    def _compute_state(
        self,
        theta_src: jnp.ndarray,
        phi_src: jnp.ndarray,
        currents_raw: jnp.ndarray,
        active: jnp.ndarray,
        V_cut: jnp.ndarray,
        include_bg_field: jnp.ndarray,
        compute_traj: jnp.ndarray,
    ):
        # Project currents to net-zero over active electrodes.
        mask = active
        I = currents_raw * mask
        n = jnp.sum(mask)
        mean = jnp.where(n > 0, jnp.sum(I) / n, 0.0)
        I = I - mean * mask

        # Electrode-driven contribution (sources/sinks).
        s = deposit_current_sources(
            self.surface,
            theta_src=theta_src,
            phi_src=phi_src,
            currents=I,
            sigma_theta=self.cfg.sigma_theta,
            sigma_phi=self.cfg.sigma_phi,
        )
        V_e, _ = solve_current_potential(
            self.surface, s, tol=self.cfg.cg_tol, maxiter=self.cfg.cg_maxiter
        )
        K_e = surface_current_from_potential(self.surface, V_e, sigma_s=self.cfg.sigma_s)

        # Cut-driven poloidal current (topological drive).
        V_cut = jnp.asarray(V_cut, dtype=jnp.float64)
        C = V_cut / self._I1
        dV_dtheta = (C / self.surface.R)  # (Nθ,1)
        K_cut = (-self.cfg.sigma_s) * (dV_dtheta / (self.surface.a * self.surface.a))[..., None] * self.surface.r_theta

        K = K_cut + K_e

        Kmag = jnp.linalg.norm(K, axis=-1)
        Ktheta = jnp.sum(K * self._e_theta, axis=-1)
        Kphi = jnp.sum(K * self._e_phi, axis=-1)

        # Visualization potential: multi-valued cut component + single-valued electrode component.
        V_vis = V_e + V_cut * self._f_theta[:, None]

        def B_fn(xyz: jnp.ndarray) -> jnp.ndarray:
            B = biot_savart_surface(self.surface, K, xyz, eps=self.cfg.biot_savart_eps)
            if float(self.cfg.Bext0) != 0.0:
                B = B + jnp.asarray(include_bg_field, dtype=B.dtype) * ideal_toroidal_field(
                    xyz, B0=float(self.cfg.Bext0), R0=float(self.cfg.R0)
                )
            return B

        n_steps = self.cfg.fieldline_steps
        n_lines = self.cfg.n_fieldlines

        def do_trace(_):
            return trace_field_lines_batch(
                B_fn,
                self._seeds,
                step_size=self.cfg.fieldline_step_size_m,
                n_steps=n_steps,
                normalize=True,
            )

        def no_trace(_):
            return jnp.zeros((n_steps + 1, n_lines, 3), dtype=jnp.float64)

        traj = jax.lax.cond(compute_traj, do_trace, no_trace, operand=None)
        return V_vis, s, Kmag, Ktheta, Kphi, traj, I

    def _apply_surface_scalar(self) -> None:
        if not self._cache:
            return

        if self.scalar_name == "|K|":
            scal = self._cache["Kmag"].reshape((-1,))
        elif self.scalar_name == "V":
            scal = self._cache["V"].reshape((-1,))
        elif self.scalar_name == "s":
            scal = self._cache["s"].reshape((-1,))
        elif self.scalar_name == "K_theta":
            scal = self._cache["Ktheta"].reshape((-1,))
        elif self.scalar_name == "K_phi":
            scal = self._cache["Kphi"].reshape((-1,))
        else:
            raise ValueError(self.scalar_name)

        self.surface_scalars_np[:] = scal.astype(np.float32, copy=False)
        self.surface_scalars_vtk.Modified()
        self.torus_poly.Modified()

        smin = float(np.nanmin(scal))
        smax = float(np.nanmax(scal))
        if not np.isfinite(smin) or not np.isfinite(smax) or smin == smax:
            smin, smax = 0.0, 1.0
        if self.scalar_name in ("s", "V", "K_theta", "K_phi"):
            vmax = max(abs(smin), abs(smax))
            smin, smax = -vmax, vmax
        self.torus_mapper.SetScalarRange(smin, smax)

    def _update_electrode_actors(self) -> None:
        Iproj = self._Iproj_cache
        if Iproj is None:
            Iproj = self._project_currents()

        for i in range(self.N):
            src, actor = self.electrode_actors[i]
            if self.active[i] <= 0.0:
                actor.SetVisibility(False)
                continue
            actor.SetVisibility(True)

            p = torus_xyz(self.cfg.R0, self.cfg.a, self.theta_src[i], self.phi_src[i])
            src.SetCenter(float(p[0]), float(p[1]), float(p[2]))

            I = float(Iproj[i])
            if i == self.selected:
                actor.GetProperty().SetColor(1.0, 0.8, 0.2)
            else:
                if I > 0:
                    actor.GetProperty().SetColor(0.85, 0.15, 0.15)
                elif I < 0:
                    actor.GetProperty().SetColor(0.15, 0.25, 0.85)
                else:
                    actor.GetProperty().SetColor(0.4, 0.4, 0.4)

            r0 = 0.04 * self.cfg.a
            r = r0 * (0.6 + 0.8 * min(abs(I) / self.cfg.current_slider_max_A, 1.0))
            src.SetRadius(float(r))
            src.Modified()

        if self.selected is not None:
            self.I_slider_rep.SetValue(float(self.currents_raw[self.selected]))

    def _update_cut_terminal_colors(self) -> None:
        # In the visualization, V jumps from ~V_cut to 0 at theta_cut.
        # "before" (red by default) is the high-potential side for V_cut>0.
        v_before = self.V_cut
        v_after = 0.0
        if v_before >= v_after:
            hi, lo = self.cut_before_actor, self.cut_after_actor
        else:
            hi, lo = self.cut_after_actor, self.cut_before_actor
        hi.GetProperty().SetColor(0.85, 0.15, 0.15)
        lo.GetProperty().SetColor(0.15, 0.25, 0.85)

    def update_solution(self) -> None:
        t0 = time.perf_counter()
        self._update_text()
        self.window.Render()

        th = jnp.asarray(self.theta_src)
        ph = jnp.asarray(self.phi_src)
        Iraw = jnp.asarray(self.currents_raw)
        act = jnp.asarray(self.active)

        V_vis, s, Kmag, Ktheta, Kphi, traj, Iproj = self._compute_jit(
            th,
            ph,
            Iraw,
            act,
            jnp.asarray(self.V_cut),
            jnp.asarray(self.include_bg_field),
            jnp.asarray(self.show_fieldlines),
        )
        V_vis.block_until_ready()
        t1 = time.perf_counter()

        self._cache = {
            "V": np.asarray(V_vis, dtype=np.float32),
            "s": np.asarray(s, dtype=np.float32),
            "Kmag": np.asarray(Kmag, dtype=np.float32),
            "Ktheta": np.asarray(Ktheta, dtype=np.float32),
            "Kphi": np.asarray(Kphi, dtype=np.float32),
        }
        self._traj_cache = np.asarray(traj, dtype=np.float32)
        self._Iproj_cache = np.asarray(Iproj, dtype=float)

        self._apply_surface_scalar()
        self._update_electrode_actors()
        self._update_cut_terminal_colors()

        self.field_actor.SetVisibility(bool(self.show_fieldlines))
        if self.show_fieldlines and self._traj_cache is not None:
            self.field_points_np[:] = self._traj_cache.reshape((-1, 3))
            self.field_points_vtk.Modified()
            self.field_poly.Modified()

        self._update_text()
        self.window.Render()
        t2 = time.perf_counter()
        print(
            "update: solve+trace {:.3f}s, total {:.3f}s  (scalar={}, active={})".format(
                t1 - t0, t2 - t0, self.scalar_name, int(np.sum(self.active))
            )
        )

    def _select_next(self) -> None:
        active_idx = np.flatnonzero(self.active > 0.0)
        if active_idx.size == 0:
            self.selected = None
            return
        if self.selected is None or self.selected not in active_idx:
            self.selected = int(active_idx[0])
            return
        k = int(np.where(active_idx == self.selected)[0][0])
        self.selected = int(active_idx[(k + 1) % active_idx.size])

    def _delete_selected(self) -> None:
        if self.selected is None:
            return
        i = self.selected
        self.active[i] = 0.0
        self.currents_raw[i] = 0.0
        self.theta_src[i] = 0.0
        self.phi_src[i] = 0.0
        self._select_next()
        self.update_solution()

    def _add_electrode(self, theta: float, phi: float, current: float) -> None:
        free = np.flatnonzero(self.active <= 0.0)
        if free.size == 0:
            print("No free electrode slots; increase n_electrodes_max.")
            return
        i = int(free[0])
        self.theta_src[i] = float(theta)
        self.phi_src[i] = float(phi)
        self.currents_raw[i] = float(current)
        self.active[i] = 1.0
        self.selected = i
        self.I_slider_rep.SetValue(float(self.currents_raw[i]))
        self.update_solution()

    def _begin_edit(self, mode: Literal["V_cut", "current"]) -> None:
        self._edit_mode = mode
        if mode == "V_cut":
            self._edit_buffer = f"{self.V_cut:.6g}"
        else:
            if self.selected is None:
                self._edit_mode = "none"
                self._edit_buffer = ""
                return
            self._edit_buffer = f"{float(self.currents_raw[self.selected]):.6g}"
        self._update_text()
        self.window.Render()

    def _handle_edit_key(self, key_sym: str, key_code: str) -> bool:
        if self._edit_mode == "none":
            return False

        if key_sym in ("Escape",):
            self._edit_mode = "none"
            self._edit_buffer = ""
            self._update_text()
            self.window.Render()
            return True

        if key_sym in ("Return", "KP_Enter"):
            try:
                val = float(self._edit_buffer.strip())
            except Exception:
                print(f"Could not parse number: {self._edit_buffer!r}")
                return True

            if self._edit_mode == "V_cut":
                self.V_cut = float(val)
                self.V_slider_rep.SetValue(float(self.V_cut))
                self._edit_mode = "none"
                self._edit_buffer = ""
                self.update_solution()
                return True

            # current
            if self.selected is not None:
                self.currents_raw[self.selected] = float(val)
                self.I_slider_rep.SetValue(float(self.currents_raw[self.selected]))
                self._edit_mode = "none"
                self._edit_buffer = ""
                self.update_solution()
            return True

        if key_sym in ("BackSpace", "Delete"):
            self._edit_buffer = self._edit_buffer[:-1]
            self._update_text()
            self.window.Render()
            return True

        if key_code and key_code in "0123456789+-eE.":
            self._edit_buffer = self._edit_buffer + key_code
            self._update_text()
            self.window.Render()
            return True

        return True

    def _on_keypress(self, _obj, _evt) -> None:
        key_sym = self.interactor.GetKeySym()
        key_code = self.interactor.GetKeyCode()

        if self._handle_edit_key(key_sym, key_code):
            return

        if key_sym in ("v", "V"):
            self._begin_edit("V_cut")
            return
        if key_sym in ("i", "I"):
            self._begin_edit("current")
            return

        if key_sym in ("a", "A"):
            self.mode = "add_source"
        elif key_sym in ("z", "Z"):
            self.mode = "add_sink"
        elif key_sym in ("m", "M"):
            self.mode = "move"
        elif key_sym in ("d", "Delete", "BackSpace"):
            self._delete_selected()
            return
        elif key_sym in ("Tab",):
            self._select_next()
            self._update_electrode_actors()
        elif key_sym in ("c", "C"):
            order: list[ScalarName] = ["|K|", "V", "s", "K_theta", "K_phi"]
            k = order.index(self.scalar_name)
            self.scalar_name = order[(k + 1) % len(order)]
            self._apply_surface_scalar()
        elif key_sym in ("f", "F"):
            self.show_fieldlines = not self.show_fieldlines
            # Need to recompute when turning on.
            if self.show_fieldlines:
                self.update_solution()
                return
            self.field_actor.SetVisibility(False)
        elif key_sym in ("b", "B"):
            self.include_bg_field = not self.include_bg_field
            print(
                f"External toroidal field (1/R): {'ON' if self.include_bg_field else 'OFF'} "
                f"(Bext0={float(self.cfg.Bext0):.3g} T at R0={self.cfg.R0:g} m)"
            )
            if self.show_fieldlines:
                self.update_solution()
                return
        elif key_sym in ("r", "R"):
            self.update_solution()
            return
        elif key_sym in ("e", "E"):
            self._export_paraview()
        elif key_sym in ("s", "S"):
            self._save_screenshot()

        self._update_text()
        self.window.Render()

    def _on_left_click(self, _obj, _evt) -> None:
        x, y = self.interactor.GetEventPosition()

        # 1) Try selecting an electrode.
        self.prop_picker.Pick(x, y, 0, self.renderer)
        actor = self.prop_picker.GetActor()
        if actor in self._electrode_actor_to_index:
            self.selected = int(self._electrode_actor_to_index[actor])
            self.I_slider_rep.SetValue(float(self.currents_raw[self.selected]))
            self._update_electrode_actors()
            self._update_text()
            self.window.Render()
            return

        # 2) Add/move electrode by picking torus surface.
        if self.mode in ("add_source", "add_sink", "move"):
            if not self.cell_picker.Pick(x, y, 0, self.renderer):
                self.mode = "none"
                self._update_text()
                self.window.Render()
                return
            p = np.array(self.cell_picker.GetPickPosition(), dtype=float)
            theta, phi = torus_angles_from_point(self.cfg.R0, p)

            if self.mode == "move":
                if self.selected is not None and self.active[self.selected] > 0:
                    self.theta_src[self.selected] = theta
                    self.phi_src[self.selected] = phi
                    self.mode = "none"
                    self.update_solution()
                else:
                    self.mode = "none"
            elif self.mode == "add_source":
                self.mode = "none"
                self._add_electrode(theta, phi, +self.cfg.current_default_A)
            elif self.mode == "add_sink":
                self.mode = "none"
                self._add_electrode(theta, phi, -self.cfg.current_default_A)
            return

        # Fall back to default camera interaction.
        self.interactor.GetInteractorStyle().OnLeftButtonDown()

    def _save_screenshot(self) -> None:
        outdir = Path("figures/gui_screenshots")
        outdir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = outdir / f"torus_cut_gui_{ts}.png"

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.window)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"Saved screenshot: {path}")

    def _export_paraview(self) -> None:
        from .paraview import fieldlines_to_vtu, point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu

        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = Path("paraview") / f"gui_torus_cut_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)

        V = self._cache.get("V")
        s = self._cache.get("s")
        Ktheta = self._cache.get("Ktheta")
        Kphi = self._cache.get("Kphi")
        Kmag = self._cache.get("Kmag")
        if V is None or s is None or Ktheta is None or Kphi is None or Kmag is None:
            print("ParaView export: no cached solution yet.")
            return

        e_theta = np.asarray(self._e_theta, dtype=float)
        e_phi = np.asarray(self._e_phi, dtype=float)
        K_vec = Ktheta[..., None] * e_theta + Kphi[..., None] * e_phi

        surf = write_vtu(
            outdir / "winding_surface.vtu",
            torus_surface_to_vtu(
                surface=self.surface,
                point_data={
                    "V": V.reshape(-1),
                    "s": s.reshape(-1),
                    "K": K_vec.reshape(-1, 3),
                    "Ktheta": Ktheta.reshape(-1),
                    "Kphi": Kphi.reshape(-1),
                    "|K|": Kmag.reshape(-1),
                    "V_cut": np.full((V.size,), float(self.V_cut), dtype=float),
                },
            ),
        )
        blocks: dict[str, str] = {"winding_surface": surf.name}

        active = np.flatnonzero(self.active > 0.0)
        if active.size > 0:
            xyz = torus_xyz(self.cfg.R0, self.cfg.a, self.theta_src[active], self.phi_src[active])
            I = (
                np.asarray(self._Iproj_cache, dtype=float)[active]
                if self._Iproj_cache is not None
                else np.asarray(self.currents_raw, dtype=float)[active]
            )
            elec = write_vtu(
                outdir / "electrodes.vtu",
                point_cloud_to_vtu(
                    points=np.asarray(xyz, dtype=float),
                    point_data={"I_A": I, "sign_I": np.sign(I)},
                ),
            )
            blocks["electrodes"] = elec.name

        # Cut ring (where the potential jump is placed), exported as a point cloud for reference.
        theta0 = float(self.cfg.theta_cut) % (2 * np.pi)
        phi_line = np.linspace(0.0, 2 * np.pi, 80, endpoint=False)
        theta_line = theta0 * np.ones_like(phi_line)
        cut_xyz = torus_xyz(self.cfg.R0, self.cfg.a, theta_line, phi_line)
        cut = write_vtu(outdir / "cut_ring.vtu", point_cloud_to_vtu(points=cut_xyz))
        blocks["cut_ring"] = cut.name

        if self.show_fieldlines and self._traj_cache is not None:
            traj_pv = np.transpose(self._traj_cache, (1, 0, 2))
            fl = write_vtu(outdir / "fieldlines.vtu", fieldlines_to_vtu(traj=traj_pv))
            blocks["fieldlines"] = fl.name

        scene = write_vtm(outdir / "scene.vtm", blocks)
        print(f"Saved ParaView scene: {scene}")

    def run(self) -> None:
        print("Starting cut+electrodes GUI. Close the window to exit.")
        self._update_text()
        self.window.Render()
        self.interactor.Initialize()
        self.interactor.Start()


def run_torus_cut_voltage_gui(
    *, cfg: CutGUIConfig = CutGUIConfig(), initial_electrodes: dict | None = None
) -> None:  # pragma: no cover
    """Entry point for the cut-voltage (+electrodes) GUI examples."""
    _require_vtk()
    print("Interactive torus cut+electrodes GUI")
    print(f"  R0={cfg.R0} a={cfg.a} n_theta={cfg.n_theta} n_phi={cfg.n_phi}")
    print(
        f"  V_cut_default={cfg.V_cut_default}  V_cut_slider_max={cfg.V_cut_slider_max}  sigma_s={cfg.sigma_s}"
    )
    print(
        f"  electrodes: n_max={cfg.n_electrodes_max}  I0={cfg.current_default_A}  Imax={cfg.current_slider_max_A}  "
        f"sigma_theta={cfg.sigma_theta} sigma_phi={cfg.sigma_phi} cg_tol={cfg.cg_tol} cg_maxiter={cfg.cg_maxiter}"
    )
    print(
        f"  n_lines={cfg.n_fieldlines} steps={cfg.fieldline_steps} ds={cfg.fieldline_step_size_m} eps={cfg.biot_savart_eps}"
    )
    print(f"  mu0={float(MU0):.6e}")
    app = TorusCutVoltageGUI(cfg, initial_electrodes=initial_electrodes)
    app.run()


class TorusVmecBnOptimizeGUI:  # pragma: no cover
    """VTK GUI: optimize electrode sources/sinks to reduce (B·n)/norm(B) on a target VMEC surface."""

    def __init__(self, cfg: VmecOptGUIConfig):
        _require_vtk()
        jax.config.update("jax_enable_x64", True)

        self.cfg = cfg
        self.surface: TorusSurface = make_torus_surface(
            R0=cfg.R0, a=cfg.a, n_theta=cfg.n_theta, n_phi=cfg.n_phi
        )

        # Target VMEC surface.
        self._target_xyz_grid, self._target_points, self._target_normals, self._target_weights = (
            self._build_target_surface()
        )

        # Background field control.
        self.B0 = float(cfg.B0)
        self.trace_include_bg = bool(cfg.trace_include_bg_default_on)

        # Scaling: I_phys = current_scale * currents_raw (then projected to net-zero over active electrodes).
        self.auto_current_scale = cfg.current_scale is None
        if cfg.current_scale is None:
            self.current_scale = self._auto_current_scale(B0=self.B0, R0=cfg.R0)
        else:
            self.current_scale = float(cfg.current_scale)

        # Optimization controls.
        self.lr = float(cfg.lr)
        self.reg_currents = float(cfg.reg_currents)
        self.sigma_s = float(cfg.sigma_s)
        self.steps_per_opt = int(cfg.steps_per_opt)
        self.optimize_positions = bool(cfg.optimize_positions)

        # Electrode state (fixed size to avoid recompilation when user adds/removes).
        self.N = int(cfg.n_electrodes_max)
        self.theta_src = np.zeros((self.N,), dtype=float)
        self.phi_src = np.zeros((self.N,), dtype=float)
        self.currents_raw = np.zeros((self.N,), dtype=float)  # dimensionless
        self.active = np.zeros((self.N,), dtype=float)

        # Initialize a random set of active electrodes.
        n0 = int(min(cfg.n_electrodes_init, cfg.n_electrodes_max))
        rng = np.random.default_rng(0)
        self.theta_src[:n0] = rng.uniform(0.0, 2 * np.pi, size=(n0,))
        self.phi_src[:n0] = rng.uniform(0.0, 2 * np.pi, size=(n0,))
        self.currents_raw[:n0] = float(cfg.init_current_raw_rms) * rng.standard_normal(size=(n0,))
        self.active[:n0] = 1.0

        self.selected: int | None = int(np.argmax(self.active)) if np.any(self.active) else None
        self.mode: Literal["none", "add_source", "add_sink", "move"] = "none"

        # Precompute unit vectors for K decomposition.
        self._e_theta = self.surface.r_theta / self.surface.a
        self._e_phi = self.surface.r_phi / jnp.sqrt(self.surface.G)[..., None]

        # Field line seeds (inside the torus).
        theta_seed = jnp.linspace(0.0, 2 * jnp.pi, cfg.n_fieldlines, endpoint=False)
        rho = 0.5 * cfg.a
        R = cfg.R0 + rho * jnp.cos(theta_seed)
        Z = rho * jnp.sin(theta_seed)
        self._seeds = jnp.stack([R, jnp.zeros_like(R), Z], axis=-1)

        self.scalar_name: ScalarName = "|K|"
        self.show_fieldlines = True

        # Cached computed state (numpy).
        self._cache: dict[str, np.ndarray] = {}
        self._traj_cache: np.ndarray | None = None
        self._Iproj_cache: np.ndarray | None = None
        self._target_Bn_over_B_cache: np.ndarray | None = None
        self._metrics_cache: dict[str, float] = {}

        # Text entry ("textbox") state.
        self._edit_mode: Literal[
            "none",
            "current",
            "B0",
            "current_scale",
            "lr",
            "steps_per_opt",
            "reg_currents",
            "sigma_s",
        ] = "none"
        self._edit_buffer: str = ""

        # Optimizer state.
        self._opt = optax.adam(self.lr)
        self._opt_state = self._opt.init(self._params_jax())

        self._build_scene()

        # Compile compute + optimization functions once.
        self._compute_jit = jax.jit(self._compute_state)
        self._opt_step_jit = jax.jit(self._opt_step)
        print("Compiling JAX pipelines (first update may take a moment)...")
        self.update_solution()

    @staticmethod
    def _auto_current_scale(*, B0: float, R0: float) -> float:
        mu0 = float(4e-7 * np.pi)
        if float(B0) == 0.0:
            return 1.0
        return float(2 * np.pi * R0 * abs(B0) / mu0)

    def _build_target_surface(self) -> tuple[np.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        cfg = self.cfg
        vmec_path = _resolve_vmec_input_path(cfg.vmec_input)
        boundary = read_vmec_boundary(str(vmec_path))
        print("VMEC target surface:")
        print(f"  file={vmec_path}  NFP={boundary.nfp}  nmodes={boundary.m.size}")

        target = vmec_target_surface(
            boundary,
            torus_R0=float(cfg.R0),
            torus_a=float(cfg.a),
            fit_margin=float(cfg.fit_margin),
            n_theta=int(cfg.surf_n_theta),
            n_phi=int(cfg.surf_n_phi),
            dtype=jnp.float64,
        )

        print("  fit into circular torus:")
        print(f"    torus: R0={cfg.R0} a={cfg.a} fit_margin={cfg.fit_margin}")
        print(
            "    shift_R={:+.6e} m  scale={:.6e}  rho_max(before)={:.6e} m".format(
                float(target.fit.shift_R), float(target.fit.scale_rho), float(target.fit.rho_max_before_m)
            )
        )

        xyz = target.xyz
        n_hat = target.normals
        w_area = target.weights

        xyz_grid_np = np.asarray(xyz, dtype=float)
        points = xyz.reshape((-1, 3))
        normals = n_hat.reshape((-1, 3))
        weights = w_area.reshape((-1,))
        return xyz_grid_np, points, normals, weights

    def _params_jax(self) -> SourceParams:
        return SourceParams(
            theta_src=jnp.asarray(self.theta_src, dtype=jnp.float64),
            phi_src=jnp.asarray(self.phi_src, dtype=jnp.float64),
            currents_raw=jnp.asarray(self.currents_raw, dtype=jnp.float64),
        )

    def _compute_state(
        self,
        theta_src: jnp.ndarray,
        phi_src: jnp.ndarray,
        currents_raw: jnp.ndarray,
        active: jnp.ndarray,
        B0: jnp.ndarray,
        current_scale: jnp.ndarray,
        sigma_s: jnp.ndarray,
        reg_currents: jnp.ndarray,
        trace_include_bg: jnp.ndarray,
        compute_traj: jnp.ndarray,
    ):
        # Project currents to net-zero over active electrodes (in raw units).
        mask = active
        Iraw = currents_raw * mask
        n = jnp.sum(mask)
        mean = jnp.where(n > 0, jnp.sum(Iraw) / n, 0.0)
        Iraw = Iraw - mean * mask
        I = current_scale * Iraw  # physical currents [A], projected

        s = deposit_current_sources(
            self.surface,
            theta_src=theta_src,
            phi_src=phi_src,
            currents=I,
            sigma_theta=self.cfg.sigma_theta,
            sigma_phi=self.cfg.sigma_phi,
        )
        V, _ = solve_current_potential(
            self.surface,
            s,
            tol=self.cfg.cg_tol,
            maxiter=self.cfg.cg_maxiter,
            use_preconditioner=self.cfg.use_preconditioner,
        )
        K = surface_current_from_potential(self.surface, V, sigma_s=sigma_s)

        Kmag = jnp.linalg.norm(K, axis=-1)
        Ktheta = jnp.sum(K * self._e_theta, axis=-1)
        Kphi = jnp.sum(K * self._e_phi, axis=-1)

        B_shell = biot_savart_surface(self.surface, K, self._target_points, eps=self.cfg.biot_savart_eps)
        B_bg = ideal_toroidal_field(self._target_points, B0=B0, R0=self.cfg.R0)
        B_tot = B_bg + B_shell
        Bn = jnp.sum(B_tot * self._target_normals, axis=-1)
        Bmag = jnp.linalg.norm(B_tot, axis=-1)
        Bn_over_B = Bn / (Bmag + 1e-30)

        w = self._target_weights
        wsum = jnp.sum(w)
        loss_bn = jnp.sum(w * (Bn_over_B * Bn_over_B)) / (wsum + 1e-30)
        loss_reg = reg_currents * jnp.mean(Iraw * Iraw)
        loss = loss_bn + loss_reg

        Bn_over_B_rms = jnp.sqrt(loss_bn)
        Bn_over_B_max = jnp.max(jnp.abs(Bn_over_B))
        I_rms = jnp.sqrt(jnp.mean(I * I))

        def B_fn(xyz: jnp.ndarray) -> jnp.ndarray:
            B = biot_savart_surface(self.surface, K, xyz, eps=self.cfg.biot_savart_eps)
            B = B + jnp.asarray(trace_include_bg, dtype=B.dtype) * ideal_toroidal_field(
                xyz, B0=B0, R0=self.cfg.R0
            )
            return B

        n_steps = self.cfg.fieldline_steps
        n_lines = self.cfg.n_fieldlines

        def do_trace(_):
            return trace_field_lines_batch(
                B_fn,
                self._seeds,
                step_size=self.cfg.fieldline_step_size_m,
                n_steps=n_steps,
                normalize=True,
            )

        def no_trace(_):
            return jnp.zeros((n_steps + 1, n_lines, 3), dtype=jnp.float64)

        traj = jax.lax.cond(compute_traj, do_trace, no_trace, operand=None)

        return (
            V,
            s,
            Kmag,
            Ktheta,
            Kphi,
            Bn_over_B,
            loss,
            loss_bn,
            loss_reg,
            Bn_over_B_rms,
            Bn_over_B_max,
            I_rms,
            traj,
            I,
        )

    def _loss_fn(
        self,
        params: SourceParams,
        *,
        active: jnp.ndarray,
        B0: jnp.ndarray,
        current_scale: jnp.ndarray,
        sigma_s: jnp.ndarray,
        reg_currents: jnp.ndarray,
    ):
        (
            _V,
            _s,
            _Kmag,
            _Ktheta,
            _Kphi,
            _Bn_over_B,
            loss,
            loss_bn,
            loss_reg,
            Bn_over_B_rms,
            Bn_over_B_max,
            I_rms,
            _traj,
            _Iproj,
        ) = (
            self._compute_state(
                params.theta_src,
                params.phi_src,
                params.currents_raw,
                active,
                B0,
                current_scale,
                sigma_s,
                reg_currents,
                jnp.asarray(True),
                jnp.asarray(False),
            )
        )
        aux = {
            "loss": loss,
            "loss_bn": loss_bn,
            "loss_reg": loss_reg,
            "Bn_over_B_rms": Bn_over_B_rms,
            "Bn_over_B_max": Bn_over_B_max,
            "I_rms_A": I_rms,
        }
        return loss, aux

    def _opt_step(
        self,
        params: SourceParams,
        opt_state,
        *,
        active: jnp.ndarray,
        B0: jnp.ndarray,
        current_scale: jnp.ndarray,
        sigma_s: jnp.ndarray,
        reg_currents: jnp.ndarray,
        optimize_positions: jnp.ndarray,
    ):
        (loss, aux), g = jax.value_and_grad(self._loss_fn, has_aux=True)(
            params,
            active=active,
            B0=B0,
            current_scale=current_scale,
            sigma_s=sigma_s,
            reg_currents=reg_currents,
        )

        theta_g = jnp.where(optimize_positions, g.theta_src, jnp.zeros_like(g.theta_src))
        phi_g = jnp.where(optimize_positions, g.phi_src, jnp.zeros_like(g.phi_src))
        g = SourceParams(theta_src=theta_g, phi_src=phi_g, currents_raw=g.currents_raw)

        updates, opt_state2 = self._opt.update(g, opt_state, params)
        p2 = optax.apply_updates(params, updates)

        twopi = 2.0 * jnp.pi
        p2 = SourceParams(
            theta_src=jnp.mod(p2.theta_src, twopi),
            phi_src=jnp.mod(p2.phi_src, twopi),
            currents_raw=p2.currents_raw,
        )
        return p2, opt_state2, aux

    def _build_scene(self) -> None:
        cfg = self.cfg

        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(*cfg.window_size)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)

        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        # Torus surface actor (with scalars updated each solve).
        xyz = np.asarray(self.surface.r)
        self.torus_poly = _build_torus_polydata(xyz)

        self.surface_scalars_np = np.zeros((cfg.n_theta * cfg.n_phi,), dtype=np.float32)
        self.surface_scalars_vtk = numpy_to_vtk(self.surface_scalars_np, deep=False)
        self.surface_scalars_vtk.SetName("scalar")
        self.torus_poly.GetPointData().SetScalars(self.surface_scalars_vtk)

        self.lut = vtk.vtkLookupTable()
        self.lut.SetNumberOfTableValues(256)
        self.lut.Build()

        self.torus_mapper = vtk.vtkPolyDataMapper()
        self.torus_mapper.SetInputData(self.torus_poly)
        self.torus_mapper.SetLookupTable(self.lut)
        self.torus_mapper.SetScalarModeToUsePointData()
        self.torus_mapper.ScalarVisibilityOn()

        self.torus_actor = vtk.vtkActor()
        self.torus_actor.SetMapper(self.torus_mapper)
        self.torus_actor.GetProperty().SetOpacity(cfg.surface_opacity)
        self.torus_actor.GetProperty().SetInterpolationToPhong()
        self.torus_actor.GetProperty().SetSpecular(0.2)
        self.torus_actor.GetProperty().SetSpecularPower(30.0)
        self.renderer.AddActor(self.torus_actor)

        # Target VMEC surface actor (colored by (B·n)/|B|).
        self.target_poly = _build_torus_polydata(self._target_xyz_grid)
        self.target_scalars_np = np.zeros((cfg.surf_n_theta * cfg.surf_n_phi,), dtype=np.float32)
        self.target_scalars_vtk = numpy_to_vtk(self.target_scalars_np, deep=False)
        self.target_scalars_vtk.SetName("Bn_over_B")
        self.target_poly.GetPointData().SetScalars(self.target_scalars_vtk)

        self.target_mapper = vtk.vtkPolyDataMapper()
        self.target_mapper.SetInputData(self.target_poly)
        self.target_mapper.SetLookupTable(self.lut)
        self.target_mapper.SetScalarModeToUsePointData()
        self.target_mapper.ScalarVisibilityOn()

        self.target_actor = vtk.vtkActor()
        self.target_actor.SetMapper(self.target_mapper)
        self.target_actor.GetProperty().SetOpacity(cfg.target_opacity)
        self.target_actor.GetProperty().SetInterpolationToPhong()
        self.target_actor.GetProperty().SetSpecular(0.1)
        self.target_actor.GetProperty().SetSpecularPower(20.0)
        self.renderer.AddActor(self.target_actor)

        # Axis curve for reference.
        axis_phi = np.linspace(0, 2 * np.pi, 200, endpoint=True)
        axis = np.stack([cfg.R0 * np.cos(axis_phi), cfg.R0 * np.sin(axis_phi), 0.0 * axis_phi], axis=-1)
        axis_poly = vtk.vtkPolyData()
        axis_points = vtk.vtkPoints()
        axis_points.SetData(numpy_to_vtk(axis, deep=True))
        axis_poly.SetPoints(axis_points)
        axis_lines = vtk.vtkCellArray()
        pl = vtk.vtkPolyLine()
        pl.GetPointIds().SetNumberOfIds(axis.shape[0])
        for i in range(axis.shape[0]):
            pl.GetPointIds().SetId(i, i)
        axis_lines.InsertNextCell(pl)
        axis_poly.SetLines(axis_lines)
        axis_mapper = vtk.vtkPolyDataMapper()
        axis_mapper.SetInputData(axis_poly)
        axis_actor = vtk.vtkActor()
        axis_actor.SetMapper(axis_mapper)
        axis_actor.GetProperty().SetColor(0.0, 0.0, 0.0)
        axis_actor.GetProperty().SetLineWidth(2.0)
        self.renderer.AddActor(axis_actor)

        # Field lines actor (updated each solve).
        n_pts_line = self.cfg.fieldline_steps + 1
        self.field_poly = _build_fieldlines_polydata(self.cfg.n_fieldlines, n_pts_line)
        self.field_points_np = np.zeros((self.cfg.n_fieldlines * n_pts_line, 3), dtype=np.float32)
        self.field_points_vtk = numpy_to_vtk(self.field_points_np, deep=False)
        self.field_points_vtk.SetName("field_points")
        self.field_poly.GetPoints().SetData(self.field_points_vtk)
        self.field_mapper = vtk.vtkPolyDataMapper()
        self.field_mapper.SetInputData(self.field_poly)
        self.field_actor = vtk.vtkActor()
        self.field_actor.SetMapper(self.field_mapper)
        self.field_actor.GetProperty().SetColor(0.1, 0.25, 0.9)
        self.field_actor.GetProperty().SetLineWidth(2.0)
        self.renderer.AddActor(self.field_actor)

        # Electrode actors (spheres).
        self.electrode_actors = []
        self._electrode_actor_to_index = {}
        for i in range(self.N):
            src = vtk.vtkSphereSource()
            src.SetThetaResolution(16)
            src.SetPhiResolution(16)
            src.SetRadius(0.05 * self.cfg.a)

            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(src.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.SetVisibility(False)
            self.renderer.AddActor(actor)

            self.electrode_actors.append((src, actor))
            self._electrode_actor_to_index[actor] = i

        # Text overlay (help + status).
        self.text = vtk.vtkTextActor()
        self.text.GetTextProperty().SetFontSize(16)
        self.text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.text.SetPosition(12, self.cfg.window_size[1] - 240)
        self.renderer.AddActor2D(self.text)

        # Editable numeric "textbox".
        self.input_text = vtk.vtkTextActor()
        self.input_text.GetTextProperty().SetFontSize(16)
        self.input_text.GetTextProperty().SetColor(0.0, 0.0, 0.0)
        self.input_text.SetPosition(12, 12)
        self.renderer.AddActor2D(self.input_text)

        # Current slider for the selected electrode (physical A).
        rep = vtk.vtkSliderRepresentation2D()
        rep.SetMinimumValue(-self.cfg.current_slider_max_A)
        rep.SetMaximumValue(+self.cfg.current_slider_max_A)
        rep.SetValue(0.0)
        rep.SetTitleText("Selected electrode current I [A]")
        rep.SetLabelFormat("%0.0f")
        rep.SetSliderLength(0.02)
        rep.SetSliderWidth(0.03)
        rep.SetTubeWidth(0.006)
        rep.SetEndCapLength(0.01)
        rep.SetEndCapWidth(0.03)
        rep.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint1Coordinate().SetValue(0.10, 0.06)
        rep.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
        rep.GetPoint2Coordinate().SetValue(0.55, 0.06)

        self.slider_rep = rep
        self.slider = vtk.vtkSliderWidget()
        self.slider.SetInteractor(self.interactor)
        self.slider.SetRepresentation(rep)
        self.slider.EnabledOn()

        def on_slider_end(_obj, _evt):
            if self.selected is None:
                return
            val_A = float(self.slider_rep.GetValue())
            if float(self.current_scale) == 0.0:
                return
            self.currents_raw[self.selected] = val_A / float(self.current_scale)
            self.update_solution()

        self.slider.AddObserver(vtk.vtkCommand.EndInteractionEvent, on_slider_end)

        # Picking helpers.
        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.0005)
        self.cell_picker.PickFromListOn()
        self.cell_picker.AddPickList(self.torus_actor)
        self.prop_picker = vtk.vtkPropPicker()

        # Interactor events.
        self.interactor.AddObserver(vtk.vtkCommand.KeyPressEvent, self._on_keypress)
        self.interactor.AddObserver(vtk.vtkCommand.LeftButtonPressEvent, self._on_left_click)

        # Initial camera.
        self.renderer.ResetCamera()

    def _help_text(self) -> str:
        return (
            "VMEC (B·n)/|B| optimization GUI (VTK)\n"
            "Goal: drive target-surface (B·n)/|B| -> 0 by moving/setting electrode sources/sinks\n"
            "Mouse: rotate/zoom as usual\n"
            "Click electrode: select\n"
            "Keys:\n"
            "  o: optimize (run N steps)\n"
            "  space: single optimization step\n"
            "  p: toggle optimize positions\n"
            "  a: add SOURCE (next click on torus)\n"
            "  z: add SINK   (next click on torus)\n"
            "  m: move selected (next click on torus)\n"
            "  d: delete selected\n"
            "  tab: cycle selected\n"
            "  c: cycle torus scalar (|K|, V, s, Kθ, Kφ)\n"
            "  f: toggle field lines\n"
            "  t: toggle include background field in field lines\n"
            "  r: recompute\n"
            "  e: export ParaView (.vtu/.vtm)\n"
            "  i (or v): type selected electrode current [A]\n"
            "  b: type background B0 [T]\n"
            "  k: type current_scale [A/unit]\n"
            "  l: type learning rate\n"
            "  n: type opt steps per 'o'\n"
            "  g: type reg_currents\n"
            "  x: type sigma_s\n"
            "  s: save screenshot\n"
        )

    def _status_text(self) -> str:
        n_active = int(np.sum(self.active))
        sel = self.selected
        if sel is None:
            sel_txt = "none"
        else:
            Iraw_A = float(self.currents_raw[sel] * float(self.current_scale))
            Iproj = None
            if self._Iproj_cache is not None and sel < self._Iproj_cache.size:
                Iproj = float(self._Iproj_cache[sel])
            if Iproj is None:
                sel_txt = f"{sel}  Iraw={Iraw_A:+.3e} A  (active={self.active[sel]:.0f})"
            else:
                sel_txt = f"{sel}  Iraw={Iraw_A:+.3e} A  Iproj={Iproj:+.3e} A  (active={self.active[sel]:.0f})"

        m = self._metrics_cache
        metric_lines = ""
        if m:
            metric_lines = (
                f"loss={m.get('loss', np.nan):.3e}  bn={m.get('loss_bn', np.nan):.3e}  reg={m.get('loss_reg', np.nan):.3e}\n"
                f"rms(Bn/B)={m.get('Bn_over_B_rms', np.nan):.3e}  max|Bn/B|={m.get('Bn_over_B_max', np.nan):.3e}  I_rms={m.get('I_rms_A', np.nan):.3e} A\n"
            )

        auto_txt = "auto" if self.auto_current_scale else "manual"
        trace_bg = "ON" if self.trace_include_bg else "OFF"
        return (
            f"Active electrodes: {n_active}/{self.N}   optimize_positions={self.optimize_positions}\n"
            f"Selected: {sel_txt}\n"
            f"Mode: {self.mode}\n"
            f"Torus scalar: {self.scalar_name}   Target scalar: (B·n)/|B|\n"
            f"B0={self.B0:.6g} T   current_scale={self.current_scale:.3e} A/unit ({auto_txt})   trace_bg={trace_bg}\n"
            f"lr={self.lr:.3e}   steps_per_opt={self.steps_per_opt}   reg_currents={self.reg_currents:.3e}   sigma_s={self.sigma_s:.3e}\n"
            + metric_lines
        )

    def _update_text(self) -> None:
        self.text.SetInput(self._help_text() + "\n" + self._status_text())

        if self._edit_mode == "none":
            self.input_text.SetInput(
                "Type: i/v=current [A], b=B0, k=current_scale, l=lr, n=steps, g=reg, x=sigma_s"
            )
        else:
            self.input_text.SetInput(
                f"Input [{self._edit_mode}]: {self._edit_buffer}   (Enter=apply, Esc=cancel)"
            )

    def _apply_surface_scalar(self) -> None:
        if not self._cache:
            return

        if self.scalar_name == "|K|":
            scal = self._cache["Kmag"].reshape((-1,))
        elif self.scalar_name == "V":
            scal = self._cache["V"].reshape((-1,))
        elif self.scalar_name == "s":
            scal = self._cache["s"].reshape((-1,))
        elif self.scalar_name == "K_theta":
            scal = self._cache["Ktheta"].reshape((-1,))
        elif self.scalar_name == "K_phi":
            scal = self._cache["Kphi"].reshape((-1,))
        else:
            raise ValueError(self.scalar_name)

        self.surface_scalars_np[:] = scal.astype(np.float32, copy=False)
        self.surface_scalars_vtk.Modified()
        self.torus_poly.Modified()

        smin = float(np.nanmin(scal))
        smax = float(np.nanmax(scal))
        if not np.isfinite(smin) or not np.isfinite(smax) or smin == smax:
            smin, smax = 0.0, 1.0
        if self.scalar_name in ("s", "K_theta", "K_phi", "V"):
            vmax = max(abs(smin), abs(smax))
            smin, smax = -vmax, vmax
        self.torus_mapper.SetScalarRange(smin, smax)

    def _apply_target_scalar(self) -> None:
        if self._target_Bn_over_B_cache is None:
            return
        scal = self._target_Bn_over_B_cache.reshape((-1,))
        self.target_scalars_np[:] = scal.astype(np.float32, copy=False)
        self.target_scalars_vtk.Modified()
        self.target_poly.Modified()

        smin = float(np.nanmin(scal))
        smax = float(np.nanmax(scal))
        if not np.isfinite(smin) or not np.isfinite(smax) or smin == smax:
            smin, smax = 0.0, 1.0
        vmax = max(abs(smin), abs(smax))
        self.target_mapper.SetScalarRange(-vmax, vmax)

    def _update_electrode_actors(self) -> None:
        Iproj = self._Iproj_cache
        if Iproj is None:
            Iproj = np.zeros_like(self.currents_raw)

        for i in range(self.N):
            src, actor = self.electrode_actors[i]
            if self.active[i] <= 0.0:
                actor.SetVisibility(False)
                continue
            actor.SetVisibility(True)

            p = torus_xyz(self.cfg.R0, self.cfg.a, self.theta_src[i], self.phi_src[i])
            src.SetCenter(float(p[0]), float(p[1]), float(p[2]))

            I = float(Iproj[i])
            if i == self.selected:
                actor.GetProperty().SetColor(1.0, 0.8, 0.2)
            else:
                if I > 0:
                    actor.GetProperty().SetColor(0.85, 0.15, 0.15)
                elif I < 0:
                    actor.GetProperty().SetColor(0.15, 0.25, 0.85)
                else:
                    actor.GetProperty().SetColor(0.4, 0.4, 0.4)

            r0 = 0.04 * self.cfg.a
            r = r0 * (0.6 + 0.8 * min(abs(I) / self.cfg.current_slider_max_A, 1.0))
            src.SetRadius(float(r))
            src.Modified()

    def update_solution(self) -> None:
        t0 = time.perf_counter()
        self._update_text()
        self.window.Render()

        th = jnp.asarray(self.theta_src, dtype=jnp.float64)
        ph = jnp.asarray(self.phi_src, dtype=jnp.float64)
        Iraw = jnp.asarray(self.currents_raw, dtype=jnp.float64)
        act = jnp.asarray(self.active, dtype=jnp.float64)

        (
            V,
            s,
            Kmag,
            Ktheta,
            Kphi,
            Bn_over_B,
            loss,
            loss_bn,
            loss_reg,
            Bn_over_B_rms,
            Bn_over_B_max,
            I_rms,
            traj,
            Iproj,
        ) = self._compute_jit(
            th,
            ph,
            Iraw,
            act,
            jnp.asarray(self.B0, dtype=jnp.float64),
            jnp.asarray(self.current_scale, dtype=jnp.float64),
            jnp.asarray(self.sigma_s, dtype=jnp.float64),
            jnp.asarray(self.reg_currents, dtype=jnp.float64),
            jnp.asarray(self.trace_include_bg),
            jnp.asarray(self.show_fieldlines),
        )
        V.block_until_ready()
        t1 = time.perf_counter()

        self._cache = {
            "V": np.asarray(V, dtype=np.float32),
            "s": np.asarray(s, dtype=np.float32),
            "Kmag": np.asarray(Kmag, dtype=np.float32),
            "Ktheta": np.asarray(Ktheta, dtype=np.float32),
            "Kphi": np.asarray(Kphi, dtype=np.float32),
        }
        self._traj_cache = np.asarray(traj, dtype=np.float32)
        self._Iproj_cache = np.asarray(Iproj, dtype=float)
        self._target_Bn_over_B_cache = np.asarray(Bn_over_B, dtype=np.float32)
        self._metrics_cache = {
            "loss": float(loss),
            "loss_bn": float(loss_bn),
            "loss_reg": float(loss_reg),
            "Bn_over_B_rms": float(Bn_over_B_rms),
            "Bn_over_B_max": float(Bn_over_B_max),
            "I_rms_A": float(I_rms),
        }

        self._apply_surface_scalar()
        self._apply_target_scalar()
        self._update_electrode_actors()

        # Update field lines geometry.
        self.field_actor.SetVisibility(bool(self.show_fieldlines))
        if self.show_fieldlines and self._traj_cache is not None:
            self.field_points_np[:] = self._traj_cache.reshape((-1, 3))
            self.field_points_vtk.Modified()
            self.field_poly.Modified()

        if self.selected is not None:
            self.slider_rep.SetValue(float(self.currents_raw[self.selected] * float(self.current_scale)))
        self._update_text()
        self.window.Render()
        t2 = time.perf_counter()
        print(
            "update: solve+trace {:.3f}s, total {:.3f}s  (scalar={}, active={})".format(
                t1 - t0, t2 - t0, self.scalar_name, int(np.sum(self.active))
            )
        )

    def _select_next(self) -> None:
        active_idx = np.flatnonzero(self.active > 0.0)
        if active_idx.size == 0:
            self.selected = None
            return
        if self.selected is None or self.selected not in active_idx:
            self.selected = int(active_idx[0])
            return
        k = int(np.where(active_idx == self.selected)[0][0])
        self.selected = int(active_idx[(k + 1) % active_idx.size])

    def _delete_selected(self) -> None:
        if self.selected is None:
            return
        i = self.selected
        self.active[i] = 0.0
        self.currents_raw[i] = 0.0
        self.theta_src[i] = 0.0
        self.phi_src[i] = 0.0
        self._select_next()
        self.update_solution()

    def _add_electrode(self, theta: float, phi: float, current_A: float) -> None:
        free = np.flatnonzero(self.active <= 0.0)
        if free.size == 0:
            print("No free electrode slots; increase n_electrodes_max.")
            return
        i = int(free[0])
        self.theta_src[i] = float(theta)
        self.phi_src[i] = float(phi)
        if float(self.current_scale) != 0.0:
            self.currents_raw[i] = float(current_A) / float(self.current_scale)
        else:
            self.currents_raw[i] = 0.0
        self.active[i] = 1.0
        self.selected = i
        self.slider_rep.SetValue(float(current_A))
        self.update_solution()

    def _begin_edit(self, mode: str) -> None:
        self._edit_mode = mode  # type: ignore[assignment]
        if mode == "current":
            if self.selected is None:
                self._edit_mode = "none"
                self._edit_buffer = ""
                return
            I_A = float(self.currents_raw[self.selected] * float(self.current_scale))
            self._edit_buffer = f"{I_A:.6g}"
        elif mode == "B0":
            self._edit_buffer = f"{self.B0:.6g}"
        elif mode == "current_scale":
            self._edit_buffer = f"{self.current_scale:.6g}"
        elif mode == "lr":
            self._edit_buffer = f"{self.lr:.6g}"
        elif mode == "steps_per_opt":
            self._edit_buffer = f"{self.steps_per_opt:d}"
        elif mode == "reg_currents":
            self._edit_buffer = f"{self.reg_currents:.6g}"
        elif mode == "sigma_s":
            self._edit_buffer = f"{self.sigma_s:.6g}"
        else:
            self._edit_mode = "none"
            self._edit_buffer = ""
            return
        self._update_text()
        self.window.Render()

    def _handle_edit_key(self, key_sym: str, key_code: str) -> bool:
        if self._edit_mode == "none":
            return False

        if key_sym in ("Escape",):
            self._edit_mode = "none"
            self._edit_buffer = ""
            self._update_text()
            self.window.Render()
            return True

        if key_sym in ("Return", "KP_Enter"):
            mode = self._edit_mode
            txt = self._edit_buffer.strip()
            try:
                if mode == "steps_per_opt":
                    val = int(float(txt))
                else:
                    val = float(txt)
            except Exception:
                print(f"Could not parse number: {self._edit_buffer!r}")
                return True

            if mode == "current" and self.selected is not None:
                if float(self.current_scale) != 0.0:
                    self.currents_raw[self.selected] = float(val) / float(self.current_scale)
                    self.slider_rep.SetValue(float(val))
            elif mode == "B0":
                self.B0 = float(val)
                if self.auto_current_scale:
                    self.current_scale = self._auto_current_scale(B0=self.B0, R0=self.cfg.R0)
            elif mode == "current_scale":
                self.current_scale = float(val)
                self.auto_current_scale = False
            elif mode == "lr":
                self.lr = float(val)
                self._opt = optax.adam(self.lr)
                self._opt_state = self._opt.init(self._params_jax())
                self._opt_step_jit = jax.jit(self._opt_step)
            elif mode == "steps_per_opt":
                self.steps_per_opt = max(1, int(val))
            elif mode == "reg_currents":
                self.reg_currents = float(val)
            elif mode == "sigma_s":
                self.sigma_s = float(val)

            self._edit_mode = "none"
            self._edit_buffer = ""
            self.update_solution()
            return True

        if key_sym in ("BackSpace", "Delete"):
            self._edit_buffer = self._edit_buffer[:-1]
            self._update_text()
            self.window.Render()
            return True

        if key_code and key_code in "0123456789+-eE.":
            self._edit_buffer = self._edit_buffer + key_code
            self._update_text()
            self.window.Render()
            return True

        return True

    def _optimize(self, n_steps: int) -> None:
        if n_steps <= 0:
            return
        params = self._params_jax()
        act = jnp.asarray(self.active, dtype=jnp.float64)
        B0 = jnp.asarray(self.B0, dtype=jnp.float64)
        current_scale = jnp.asarray(self.current_scale, dtype=jnp.float64)
        sigma_s = jnp.asarray(self.sigma_s, dtype=jnp.float64)
        reg = jnp.asarray(self.reg_currents, dtype=jnp.float64)
        opt_pos = jnp.asarray(self.optimize_positions)

        t0 = time.perf_counter()
        aux_last = None
        for _k in range(int(n_steps)):
            params, self._opt_state, aux = self._opt_step_jit(
                params,
                self._opt_state,
                active=act,
                B0=B0,
                current_scale=current_scale,
                sigma_s=sigma_s,
                reg_currents=reg,
                optimize_positions=opt_pos,
            )
            aux_last = aux
        params.theta_src.block_until_ready()
        t1 = time.perf_counter()

        self.theta_src[:] = np.asarray(params.theta_src)
        self.phi_src[:] = np.asarray(params.phi_src)
        self.currents_raw[:] = np.asarray(params.currents_raw)

        if aux_last is not None:
            print(
                "optimize: steps={}  loss={:.3e}  rms(Bn/B)={:.3e}  max|Bn/B|={:.3e}  I_rms={:.3e}A  wall={:.3f}s".format(
                    int(n_steps),
                    float(aux_last["loss"]),
                    float(aux_last["Bn_over_B_rms"]),
                    float(aux_last["Bn_over_B_max"]),
                    float(aux_last["I_rms_A"]),
                    t1 - t0,
                )
            )
        self.update_solution()

    def _on_keypress(self, _obj, _evt) -> None:
        key_sym = self.interactor.GetKeySym()
        key_code = self.interactor.GetKeyCode()

        if self._handle_edit_key(key_sym, key_code):
            return

        if key_sym in ("i", "I", "v", "V"):
            self._begin_edit("current")
            return
        if key_sym in ("b", "B"):
            self._begin_edit("B0")
            return
        if key_sym in ("k", "K"):
            self._begin_edit("current_scale")
            return
        if key_sym in ("l", "L"):
            self._begin_edit("lr")
            return
        if key_sym in ("n", "N"):
            self._begin_edit("steps_per_opt")
            return
        if key_sym in ("g", "G"):
            self._begin_edit("reg_currents")
            return
        if key_sym in ("x", "X"):
            self._begin_edit("sigma_s")
            return

        if key_sym in ("space",):
            self._optimize(1)
            return

        if key_sym in ("o", "O"):
            self._optimize(self.steps_per_opt)
            return

        if key_sym in ("p", "P"):
            self.optimize_positions = not self.optimize_positions

        if key_sym in ("a", "A"):
            self.mode = "add_source"
        elif key_sym in ("z", "Z"):
            self.mode = "add_sink"
        elif key_sym in ("m", "M"):
            self.mode = "move"
        elif key_sym in ("d", "Delete", "BackSpace"):
            self._delete_selected()
            return
        elif key_sym in ("Tab",):
            self._select_next()
            self._update_electrode_actors()
        elif key_sym in ("c", "C"):
            order: list[ScalarName] = ["|K|", "V", "s", "K_theta", "K_phi"]
            k = order.index(self.scalar_name)
            self.scalar_name = order[(k + 1) % len(order)]
            self._apply_surface_scalar()
        elif key_sym in ("f", "F"):
            self.show_fieldlines = not self.show_fieldlines
            if self.show_fieldlines:
                self.update_solution()
                return
            self.field_actor.SetVisibility(False)
        elif key_sym in ("t", "T"):
            self.trace_include_bg = not self.trace_include_bg
            print(f"Fieldline tracing background field: {'ON' if self.trace_include_bg else 'OFF'} (B0={self.B0:.6g} T)")
            if self.show_fieldlines:
                self.update_solution()
                return
        elif key_sym in ("r", "R"):
            self.update_solution()
            return
        elif key_sym in ("e", "E"):
            self._export_paraview()
        elif key_sym in ("s", "S"):
            self._save_screenshot()

        self._update_text()
        self.window.Render()

    def _on_left_click(self, _obj, _evt) -> None:
        x, y = self.interactor.GetEventPosition()

        # 1) Try selecting an electrode.
        self.prop_picker.Pick(x, y, 0, self.renderer)
        actor = self.prop_picker.GetActor()
        if actor in self._electrode_actor_to_index:
            self.selected = int(self._electrode_actor_to_index[actor])
            self.slider_rep.SetValue(float(self.currents_raw[self.selected] * float(self.current_scale)))
            self._update_electrode_actors()
            self._update_text()
            self.window.Render()
            return

        # 2) Add/move electrode by picking the torus surface.
        if self.mode in ("add_source", "add_sink", "move"):
            if not self.cell_picker.Pick(x, y, 0, self.renderer):
                self.mode = "none"
                self._update_text()
                self.window.Render()
                return
            p = np.array(self.cell_picker.GetPickPosition(), dtype=float)
            theta, phi = torus_angles_from_point(self.cfg.R0, p)

            if self.mode == "move":
                if self.selected is not None and self.active[self.selected] > 0:
                    self.theta_src[self.selected] = theta
                    self.phi_src[self.selected] = phi
                    self.mode = "none"
                    self.update_solution()
                else:
                    self.mode = "none"
            elif self.mode == "add_source":
                self.mode = "none"
                self._add_electrode(theta, phi, +self.cfg.current_default_A)
            elif self.mode == "add_sink":
                self.mode = "none"
                self._add_electrode(theta, phi, -self.cfg.current_default_A)
            return

        self.interactor.GetInteractorStyle().OnLeftButtonDown()

    def _save_screenshot(self) -> None:
        outdir = Path("figures/gui_screenshots")
        outdir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = outdir / f"torus_vmec_opt_gui_{ts}.png"

        w2i = vtk.vtkWindowToImageFilter()
        w2i.SetInput(self.window)
        w2i.Update()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(str(path))
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        print(f"Saved screenshot: {path}")

    def _export_paraview(self) -> None:
        from .paraview import fieldlines_to_vtu, point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu

        ts = time.strftime("%Y%m%d_%H%M%S")
        outdir = Path("paraview") / f"gui_torus_vmec_opt_{ts}"
        outdir.mkdir(parents=True, exist_ok=True)

        V = self._cache.get("V")
        s = self._cache.get("s")
        Ktheta = self._cache.get("Ktheta")
        Kphi = self._cache.get("Kphi")
        Kmag = self._cache.get("Kmag")
        if V is None or s is None or Ktheta is None or Kphi is None or Kmag is None:
            print("ParaView export: no cached winding-surface solution yet.")
            return

        e_theta = np.asarray(self._e_theta, dtype=float)
        e_phi = np.asarray(self._e_phi, dtype=float)
        K_vec = Ktheta[..., None] * e_theta + Kphi[..., None] * e_phi

        surf = write_vtu(
            outdir / "winding_surface.vtu",
            torus_surface_to_vtu(
                surface=self.surface,
                point_data={
                    "V": V.reshape(-1),
                    "s": s.reshape(-1),
                    "K": K_vec.reshape(-1, 3),
                    "Ktheta": Ktheta.reshape(-1),
                    "Kphi": Kphi.reshape(-1),
                    "|K|": Kmag.reshape(-1),
                },
            ),
        )

        blocks: dict[str, str] = {"winding_surface": surf.name}

        # Target points (VMEC surface) with Bn/B scalar if available.
        tgt_pts = np.asarray(self._target_points, dtype=float)
        pd = {
            "n_hat": np.asarray(self._target_normals, dtype=float),
            "weight": np.asarray(self._target_weights, dtype=float),
        }
        if self._target_Bn_over_B_cache is not None:
            pd["Bn_over_B"] = np.asarray(self._target_Bn_over_B_cache, dtype=float)
        tgt = write_vtu(outdir / "target_points.vtu", point_cloud_to_vtu(points=tgt_pts, point_data=pd))
        blocks["target_points"] = tgt.name

        active = np.flatnonzero(self.active > 0.0)
        if active.size > 0:
            xyz = torus_xyz(self.cfg.R0, self.cfg.a, self.theta_src[active], self.phi_src[active])
            I = (
                np.asarray(self._Iproj_cache, dtype=float)[active]
                if self._Iproj_cache is not None
                else np.asarray(self.currents_raw, dtype=float)[active] * float(self.current_scale)
            )
            elec = write_vtu(
                outdir / "electrodes.vtu",
                point_cloud_to_vtu(
                    points=np.asarray(xyz, dtype=float),
                    point_data={"I_A": I, "sign_I": np.sign(I)},
                ),
            )
            blocks["electrodes"] = elec.name

        if self.show_fieldlines and self._traj_cache is not None:
            traj_pv = np.transpose(self._traj_cache, (1, 0, 2))
            fl = write_vtu(outdir / "fieldlines.vtu", fieldlines_to_vtu(traj=traj_pv))
            blocks["fieldlines"] = fl.name

        scene = write_vtm(outdir / "scene.vtm", blocks)
        print(f"Saved ParaView scene: {scene}")

    def run(self) -> None:
        print("Starting VMEC (B·n)/|B| optimization GUI. Close the window to exit.")
        self._update_text()
        self.window.Render()
        self.interactor.Initialize()
        self.interactor.Start()


def run_torus_vmec_optimize_gui(*, cfg: VmecOptGUIConfig = VmecOptGUIConfig()) -> None:  # pragma: no cover
    _require_vtk()
    print("Interactive torus VMEC (B·n)/|B| optimization GUI")
    print(f"  R0={cfg.R0} a={cfg.a} n_theta={cfg.n_theta} n_phi={cfg.n_phi}")
    print(f"  target: vmec_input={cfg.vmec_input} surf_n_theta={cfg.surf_n_theta} surf_n_phi={cfg.surf_n_phi}")
    print(f"  B0={cfg.B0}T  sigma_theta={cfg.sigma_theta} sigma_phi={cfg.sigma_phi} sigma_s={cfg.sigma_s}")
    print(f"  opt: lr={cfg.lr} reg_currents={cfg.reg_currents} steps_per_opt={cfg.steps_per_opt}")
    print(f"  electrodes: init={cfg.n_electrodes_init}/{cfg.n_electrodes_max} init_current_raw_rms={cfg.init_current_raw_rms}")
    print(f"  mu0={float(MU0):.6e}")
    app = TorusVmecBnOptimizeGUI(cfg)
    app.run()
