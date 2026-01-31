from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping


def _init_matplotlib_config() -> None:
    # Matplotlib/fontconfig caches often default to non-writable locations in sandboxes.
    # Setting MPLCONFIGDIR prevents repeated cache rebuilds and noisy warnings.
    mpl_dir = Path(os.environ.get("MPLCONFIGDIR", "")) if os.environ.get("MPLCONFIGDIR") else None
    if mpl_dir is None or not str(mpl_dir):
        mpl_dir = Path(tempfile.gettempdir()) / "torus_solver_mplconfig"
        os.environ["MPLCONFIGDIR"] = str(mpl_dir)
    mpl_dir.mkdir(parents=True, exist_ok=True)

    # Help fontconfig find a writable cache (common issue on locked-down systems).
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", "")) if os.environ.get("XDG_CACHE_HOME") else None
    if xdg_cache is None or not str(xdg_cache):
        xdg_cache = Path(tempfile.gettempdir()) / "torus_solver_cache"
        os.environ["XDG_CACHE_HOME"] = str(xdg_cache)
    xdg_cache.mkdir(parents=True, exist_ok=True)


_init_matplotlib_config()

import matplotlib  # noqa: E402  (after MPLCONFIGDIR)

# Force a non-interactive backend so examples work on headless systems.
matplotlib.use("Agg", force=True)  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def fix_matplotlib_3d(ax) -> None:
    """Equal aspect ratio for Matplotlib 3D axes.

    Matplotlib's 3D projection does not guarantee equal scaling by default.
    This helper enforces equal data ranges in x/y/z so that geometric objects
    (e.g. circles) look undistorted.

    The implementation follows a common pattern:

    - Compute midpoints and ranges of (xlim, ylim, zlim)
    - Set symmetric limits with the same half-range on all axes
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_mid = 0.5 * (x_limits[0] + x_limits[1])
    y_range = abs(y_limits[1] - y_limits[0])
    y_mid = 0.5 * (y_limits[0] + y_limits[1])
    z_range = abs(z_limits[1] - z_limits[0])
    z_mid = 0.5 * (z_limits[0] + z_limits[1])

    R = 0.5 * max(x_range, y_range, z_range, 1e-12)
    ax.set_xlim3d([x_mid - R, x_mid + R])
    ax.set_ylim3d([y_mid - R, y_mid + R])
    ax.set_zlim3d([z_mid - R, z_mid + R])

    # Newer Matplotlib versions support a true "box aspect".
    try:
        ax.set_box_aspect((1.0, 1.0, 1.0))
    except Exception:
        pass


def set_plot_style(*, small: bool = False) -> None:
    """Set consistent, publication-style Matplotlib defaults."""
    base = 11.0 if not small else 9.5
    plt.rcParams.update(
        {
            "font.size": base,
            "axes.labelsize": base + 1,
            "axes.titlesize": base + 2,
            "legend.fontsize": base - 1,
            "xtick.labelsize": base - 1,
            "ytick.labelsize": base - 1,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.linewidth": 1.0,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "figure.figsize": (7.0, 4.5) if not small else (6.2, 4.0),
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "legend.frameon": False,
        }
    )


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig: plt.Figure, path: str | Path, *, dpi: int = 300) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _as_numpy(x):
    return np.asarray(x)


def plot_loss_history(
    *,
    steps: Iterable[int],
    metrics: Mapping[str, Iterable[float]],
    title: str,
    path: str | Path,
    yscale: str = "log",
) -> None:
    steps = np.asarray(list(steps), dtype=int)
    fig, ax = plt.subplots(constrained_layout=True)

    for name, series in metrics.items():
        y = np.asarray(list(series), dtype=float)
        ax.plot(steps, y, label=name)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Metric")
    if yscale == "log":
        min_pos = np.inf
        any_nonpos = False
        for series in metrics.values():
            y = np.asarray(list(series), dtype=float)
            any_nonpos = any_nonpos or np.any(y <= 0)
            if np.any(y > 0):
                min_pos = min(min_pos, float(np.min(y[y > 0])))
        if not np.isfinite(min_pos):
            ax.set_yscale("linear")
        elif any_nonpos:
            ax.set_yscale("symlog", linthresh=min_pos)
        else:
            ax.set_yscale("log")
    else:
        ax.set_yscale(yscale)
    ax.legend(ncol=2)
    savefig(fig, path)


def plot_surface_map(
    *,
    phi: np.ndarray,
    theta: np.ndarray,
    data: np.ndarray,
    title: str,
    cbar_label: str,
    path: str | Path,
    cmap: str = "viridis",
    vmin=None,
    vmax=None,
    overlay_points: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    """Plot a 2D map on (φ, θ) with optional point overlay."""
    phi = _as_numpy(phi)
    theta = _as_numpy(theta)
    data = _as_numpy(data)

    fig, ax = plt.subplots(constrained_layout=True)

    # Prefer full-period extents for periodic grids (common for torus parameterizations).
    def _periodic_extent(x: np.ndarray) -> tuple[float, float]:
        if x.ndim == 1 and x.size >= 2:
            dx = x[1] - x[0]
            if np.allclose(np.diff(x), dx, rtol=0, atol=1e-12):
                return float(x[0]), float(x[0] + dx * x.size)
        return float(x.min()), float(x.max())

    x0, x1 = _periodic_extent(phi)
    y0, y1 = _periodic_extent(theta)

    im = ax.imshow(
        data,
        origin="lower",
        aspect="equal",
        extent=(x0, x1, y0, y1),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\phi$ [rad]")
    ax.set_ylabel(r"$\theta$ [rad]")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    if overlay_points is not None:
        phi_p, theta_p = overlay_points
        ax.scatter(_as_numpy(phi_p), _as_numpy(theta_p), s=35, c="w", ec="k", lw=0.8, zorder=5)

    savefig(fig, path)


def plot_axis_field_comparison(
    *,
    phi: np.ndarray,
    B_target: np.ndarray,
    B_init: np.ndarray,
    B_final: np.ndarray,
    basis: tuple[np.ndarray, np.ndarray, np.ndarray],  # (e_r, e_phi, e_z)
    path: str | Path,
    title: str,
) -> None:
    """Compare B components on the axis (or any curve parameterized by phi)."""
    phi = _as_numpy(phi)
    B_target = _as_numpy(B_target)
    B_init = _as_numpy(B_init)
    B_final = _as_numpy(B_final)
    e_r, e_phi, e_z = map(_as_numpy, basis)

    def comps(B):
        Br = np.sum(B * e_r, axis=-1)
        Bp = np.sum(B * e_phi, axis=-1)
        Bz = np.sum(B * e_z, axis=-1)
        return Br, Bp, Bz

    Br_t, Bp_t, Bz_t = comps(B_target)
    Br_0, Bp_0, Bz_0 = comps(B_init)
    Br_1, Bp_1, Bz_1 = comps(B_final)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7.0, 7.5), constrained_layout=True)
    for ax, comp_name, t, b0, b1 in [
        (axs[0], r"$B_r$", Br_t, Br_0, Br_1),
        (axs[1], r"$B_\phi$", Bp_t, Bp_0, Bp_1),
        (axs[2], r"$B_z$", Bz_t, Bz_0, Bz_1),
    ]:
        ax.plot(phi, t, "k--", label="target")
        ax.plot(phi, b0, color="tab:blue", label="init")
        ax.plot(phi, b1, color="tab:orange", label="final")
        ax.set_ylabel(comp_name + " [T]")
        ax.legend(loc="best")

    axs[-1].set_xlabel(r"$\phi$ [rad]")
    fig.suptitle(title)
    savefig(fig, path)


def plot_3d_torus(
    *,
    torus_xyz: np.ndarray,  # (Nθ,Nφ,3)
    path: str | Path,
    title: str,
    electrodes_xyz: np.ndarray | None = None,
    curve_xyz: np.ndarray | None = None,
    eval_xyz: np.ndarray | None = None,
    stride: int = 3,
) -> None:
    """3D view of the torus plus optional points/curves."""
    torus_xyz = _as_numpy(torus_xyz)
    X = torus_xyz[::stride, ::stride, 0]
    Y = torus_xyz[::stride, ::stride, 1]
    Z = torus_xyz[::stride, ::stride, 2]

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color="0.75", linewidth=0.6)

    if curve_xyz is not None:
        c = _as_numpy(curve_xyz)
        ax.plot(c[:, 0], c[:, 1], c[:, 2], color="k", lw=2.0, label="curve")

    if eval_xyz is not None:
        p = _as_numpy(eval_xyz)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=12, c="tab:blue", alpha=0.6, label="eval")

    if electrodes_xyz is not None:
        e = _as_numpy(electrodes_xyz)
        ax.scatter(e[:, 0], e[:, 1], e[:, 2], s=45, c="tab:red", ec="k", lw=0.8, label="electrodes")

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="upper left")

    fix_matplotlib_3d(ax)

    ax.view_init(elev=25, azim=35)
    savefig(fig, path)


def plot_fieldline(
    *,
    pts: np.ndarray,  # (N,3)
    path3d: str | Path,
    path_rz: str | Path,
    title: str,
    R0: float | None = None,
) -> None:
    pts = _as_numpy(pts)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    R = np.sqrt(x * x + y * y)

    fig = plt.figure(figsize=(7.0, 6.0))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="tab:blue", lw=1.6)
    ax.scatter([x[0]], [y[0]], [z[0]], c="k", s=30, label="start")
    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.legend(loc="best")
    ax.view_init(elev=25, azim=35)
    fix_matplotlib_3d(ax)
    savefig(fig, path3d)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(R, z, color="tab:blue", lw=1.8)
    ax.scatter([R[0]], [z[0]], c="k", s=25)
    if R0 is not None:
        ax.axvline(R0, color="0.5", lw=1.0, ls="--")
    ax.set_title(title + " (poloidal projection)")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Z [m]")
    ax.axis("equal")
    savefig(fig, path_rz)


def plot_fieldlines_3d(
    *,
    torus_xyz: np.ndarray,  # (Nθ,Nφ,3)
    traj: np.ndarray,  # (n_steps+1, n_lines, 3)
    path: str | Path,
    title: str,
    stride: int = 3,
    line_stride: int = 1,
) -> None:
    """3D view of the torus plus multiple field lines."""
    torus_xyz = _as_numpy(torus_xyz)
    traj = _as_numpy(traj)
    if traj.ndim != 3 or traj.shape[-1] != 3:
        raise ValueError(f"Expected traj shape (n_steps+1, n_lines, 3), got {traj.shape}")

    X = torus_xyz[::stride, ::stride, 0]
    Y = torus_xyz[::stride, ::stride, 1]
    Z = torus_xyz[::stride, ::stride, 2]

    fig = plt.figure(figsize=(7.4, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color="0.8", linewidth=0.6)

    n_lines = traj.shape[1]
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(n_lines, 2)))
    for i in range(n_lines):
        pts = traj[::line_stride, i, :]
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color=colors[i % colors.shape[0]], lw=1.6, alpha=0.95)
        ax.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], s=10, c=[colors[i % colors.shape[0]]], alpha=0.9)

    ax.set_title(title)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")

    fix_matplotlib_3d(ax)

    ax.view_init(elev=25, azim=35)
    savefig(fig, path)
