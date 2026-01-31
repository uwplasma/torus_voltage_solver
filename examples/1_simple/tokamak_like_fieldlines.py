#!/usr/bin/env python3
"""Example: trace field lines in an analytic tokamak-like 1/R field.

Run:
  - `python examples/1_simple/tokamak_like_fieldlines.py`
"""

from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    sys.path.insert(0, str(root / "src"))

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from torus_solver import make_torus_surface
from torus_solver.fieldline import trace_field_line
from torus_solver.fields import toroidal_poloidal_coords, tokamak_like_field
from torus_solver.paraview import fieldlines_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu
from torus_solver.plotting import ensure_dir, plot_fieldline, savefig, set_plot_style
import torus_solver.plotting as tplot


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0)
    p.add_argument("--a0", type=float, default=0.4, help="Minor-radius offset for initial point (m)")
    p.add_argument("--Btor0", type=float, default=1.0, help="Toroidal field at R=R0 (T)")
    p.add_argument("--Bpol0", type=float, default=0.1, help="Poloidal field at R=R0 (T)")
    p.add_argument("--step", type=float, default=0.02)
    p.add_argument("--n", type=int, default=4000)
    p.add_argument("--outdir", type=str, default="figures/tokamak_like_fieldlines")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true", help="Disable ParaView (.vtu/.vtm) outputs")
    args = p.parse_args()

    if not args.no_plots:
        set_plot_style()
        outdir = ensure_dir(args.outdir)
        ensure_dir(outdir / "traj")
        ensure_dir(outdir / "angles")
        ensure_dir(outdir / "field")
        print(f"Saving figures to: {outdir}")

    print("Tokamak-like analytic field")
    print(f"  R0={args.R0}  Btor0={args.Btor0}T  Bpol0={args.Bpol0}T")

    # Start slightly off the axis, at φ=0 and poloidal angle θ=0.
    r0 = jnp.array([args.R0 + args.a0, 0.0, 0.0], dtype=jnp.float64)

    def B_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        return tokamak_like_field(xyz, B_tor0=args.Btor0, B_pol0=args.Bpol0, R0=args.R0)

    tracer = jax.jit(lambda x: trace_field_line(B_fn, x, step_size=args.step, n_steps=args.n))
    pts = tracer(r0)

    R, phi, Z, rho, theta, *_ = toroidal_poloidal_coords(pts, R0=args.R0)

    dphi = wrap_angle(phi[-1] - phi[0])
    dtheta = wrap_angle(theta[-1] - theta[0])

    print("Trace summary")
    print(f"  n_steps={args.n} step={args.step} normalize=True")
    print(f"  R range:   [{float(jnp.min(R)):.6f}, {float(jnp.max(R)):.6f}] m")
    print(f"  Z range:   [{float(jnp.min(Z)):.6f}, {float(jnp.max(Z)):.6f}] m")
    print(f"  ρ range:   [{float(jnp.min(rho)):.6f}, {float(jnp.max(rho)):.6f}] m")
    print(f"  Δφ mod 2π: {float(dphi):+.6f} rad")
    print(f"  Δθ mod 2π: {float(dtheta):+.6f} rad")

    # A crude "pitch" diagnostic:
    #   if we get several toroidal turns, estimate Δθ/Δφ from unwrapped angles.
    phi_u = jnp.unwrap(phi)
    theta_u = jnp.unwrap(theta)
    slope = (theta_u[-1] - theta_u[0]) / (phi_u[-1] - phi_u[0] + 1e-30)
    print(f"  approx dθ/dφ ≈ {float(slope):.4f}")

    if not args.no_paraview:
        outdir_p = ensure_dir(args.outdir)
        pv_dir = ensure_dir(outdir_p / "paraview")

        pts_np = np.asarray(pts, dtype=float)
        B_np = np.asarray(tokamak_like_field(pts, B_tor0=args.Btor0, B_pol0=args.Bpol0, R0=args.R0))
        Bmag = np.linalg.norm(B_np, axis=-1)

        line_vtu = write_vtu(
            pv_dir / "fieldline.vtu",
            fieldlines_to_vtu(
                traj=pts_np[None, :, :],
                point_data={
                    "B": B_np,
                    "|B|": Bmag,
                    "phi_unwrapped": np.asarray(phi_u, dtype=float),
                    "theta_unwrapped": np.asarray(theta_u, dtype=float),
                    "rho": np.asarray(rho, dtype=float),
                },
            ),
        )

        # A convenient torus surface for context: choose a minor radius that encloses the traced line.
        rho_max = float(jnp.max(rho))
        a_vis = float(max(1.2 * rho_max, 1e-3))
        surf = make_torus_surface(R0=float(args.R0), a=a_vis, n_theta=48, n_phi=48)
        surf_vtu = write_vtu(pv_dir / "torus_context.vtu", torus_surface_to_vtu(surface=surf))

        scene = write_vtm(pv_dir / "scene.vtm", {"torus_context": surf_vtu.name, "fieldline": line_vtu.name})
        print(f"ParaView scene: {scene}")

    if not args.no_plots:
        pts_np = np.asarray(pts)
        plot_fieldline(
            pts=pts_np,
            title="Tokamak-like analytic field line",
            R0=args.R0,
            path3d=outdir / "traj" / "fieldline_3d.png",
            path_rz=outdir / "traj" / "fieldline_RZ.png",
        )

        # Angle evolution.
        s = np.arange(pts_np.shape[0]) * args.step
        fig, ax = tplot.plt.subplots(constrained_layout=True)
        ax.plot(s, np.asarray(phi_u), label=r"$\phi$ (unwrapped)")
        ax.plot(s, np.asarray(theta_u), label=r"$\theta$ (unwrapped)")
        ax.set_xlabel("s (step*ds) [arb.]")
        ax.set_ylabel("Angle [rad]")
        ax.set_title("Unwrapped angles along the traced line")
        ax.legend()
        savefig(fig, outdir / "angles" / "angles_vs_s.png", dpi=args.dpi)

        fig, ax = tplot.plt.subplots(constrained_layout=True)
        ax.plot(np.asarray(phi_u), np.asarray(theta_u), color="tab:blue", lw=1.2)
        ax.set_xlabel(r"$\phi$ (unwrapped) [rad]")
        ax.set_ylabel(r"$\theta$ (unwrapped) [rad]")
        ax.set_title(r"Field-line pitch: $\theta$ vs $\phi$")
        savefig(fig, outdir / "angles" / "theta_vs_phi.png", dpi=args.dpi)

        # Field magnitude along the line.
        B = np.asarray(tokamak_like_field(pts, B_tor0=args.Btor0, B_pol0=args.Bpol0, R0=args.R0))
        Bmag = np.linalg.norm(B, axis=-1)
        fig, ax = tplot.plt.subplots(constrained_layout=True)
        ax.plot(s, Bmag, color="tab:orange")
        ax.set_xlabel("s (step*ds) [arb.]")
        ax.set_ylabel(r"$|B|$ [T]")
        ax.set_title("Magnetic field magnitude along the traced line")
        savefig(fig, outdir / "field" / "Bmag_vs_s.png", dpi=args.dpi)


def wrap_angle(x: jnp.ndarray) -> jnp.ndarray:
    twopi = 2.0 * jnp.pi
    return jnp.mod(x + jnp.pi, twopi) - jnp.pi


if __name__ == "__main__":
    main()
