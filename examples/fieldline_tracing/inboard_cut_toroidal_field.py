#!/usr/bin/env python3
"""Example: toroidal cut (voltage drop) on the inboard side -> poloidal current -> toroidal ~1/R field.

Physics idea:
  - A perfectly conducting torus cannot sustain a net poloidal loop current from a
    single-valued scalar potential: you need a *toroidal cut* so the potential can
    be multi-valued (a voltage jump across the cut).
  - For an axisymmetric cut-driven solution with no injected/extracted current on the
    surface (s=0), Laplace–Beltrami reduces to:
        ∂θ (R ∂θ V) = 0   =>   ∂θ V = C / R(θ),
    where R(θ) = R0 + a cosθ.
  - The voltage drop across the cut is:
        V_cut = ∮ ∂θ V dθ = C ∮ dθ / R(θ),
    which fixes C.
  - The surface current is then:
        K = -σ_s ∇_s V = -σ_s (∂θ V / a^2) r_θ
    (purely poloidal, producing a toroidal magnetic field inside).

We compute K on a grid, evaluate B via Biot–Savart at interior points, and compare the
toroidal component against the ideal 1/R scaling shape.

Run (from `torus_solver/`):
  python examples/fieldline_tracing/inboard_cut_toroidal_field.py --trace
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

from torus_solver import biot_savart_surface, ideal_toroidal_field, make_torus_surface
from torus_solver.fieldline import trace_field_line
from torus_solver.fields import cylindrical_coords, cylindrical_unit_vectors
from torus_solver.paraview import fieldlines_to_vtu, point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu
from torus_solver.plotting import ensure_dir, plot_3d_torus, plot_fieldline, plot_surface_map, savefig, set_plot_style
import torus_solver.plotting as tplot


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _cut_phase_theta(
    *,
    theta: np.ndarray,
    R0: float,
    a: float,
    theta_cut: float,
) -> np.ndarray:
    """Return a phase f(θ) in [0,1) with a jump at theta_cut (for visualizing V)."""
    theta = np.asarray(theta, dtype=float)
    theta_cut = float(theta_cut) % (2 * np.pi)

    # Index on the grid nearest the requested cut location.
    diff = _wrap_angle_np(theta - theta_cut)
    k0 = int(np.argmin(np.abs(diff)))

    R = R0 + a * np.cos(theta)
    q = 1.0 / R  # proportional to ∂θV up to a constant

    # Build a cumulative integral that starts at the cut (f=0 at theta_cut index).
    q_roll = np.roll(q, -k0)
    dtheta = float(2 * np.pi / theta.size)
    total = float(np.sum(q_roll) * dtheta)
    cum = np.concatenate([[0.0], np.cumsum(q_roll[:-1])]) * dtheta
    f_roll = cum / total
    f = np.roll(f_roll, k0)
    return f


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0)
    p.add_argument("--a", type=float, default=0.3)
    p.add_argument("--n-theta", type=int, default=96)
    p.add_argument("--n-phi", type=int, default=128)
    p.add_argument("--theta-cut", type=float, default=float(np.pi), help="Where the cut is placed (for plotting V)")
    p.add_argument("--Vcut", type=float, default=1.0, help="Voltage drop across the cut [arb]")
    p.add_argument("--sigma-s", type=float, default=1.0, help="Surface conductivity scale (sets K magnitude)")
    p.add_argument("--eps", type=float, default=1e-9, help="Biot–Savart softening [m]")
    p.add_argument("--n-eval", type=int, default=41, help="Interior evaluation points along midplane")
    p.add_argument("--rho-eval", type=float, default=0.6, help="Eval radius fraction (0..1) of minor radius")
    p.add_argument("--trace", action="store_true", help="Also trace an ideal-toroidal field line and sample B along it")
    p.add_argument("--trace-steps", type=int, default=900)
    p.add_argument("--trace-ds", type=float, default=0.01)
    p.add_argument("--outdir", type=str, default="figures/inboard_cut_toroidal_field")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true", help="Disable ParaView (.vtu/.vtm) outputs")
    p.add_argument("--plot-stride", type=int, default=3)
    args = p.parse_args()

    if not (0.0 < args.rho_eval <= 0.99):
        raise SystemExit("--rho-eval must be in (0,1).")

    if not args.no_plots:
        set_plot_style()
        outdir = ensure_dir(args.outdir)
        ensure_dir(outdir / "surface")
        ensure_dir(outdir / "field")
        ensure_dir(outdir / "geometry")
        ensure_dir(outdir / "traj")
        print(f"Saving figures to: {outdir}")

    surface = make_torus_surface(R0=args.R0, a=args.a, n_theta=args.n_theta, n_phi=args.n_phi)

    # Axisymmetric cut-driven solution: ∂θV = C/R, with C fixed by Vcut.
    Rtheta = surface.R[:, 0]
    I1 = jnp.sum((1.0 / Rtheta) * surface.dtheta)  # ∮ dθ / R(θ)
    C = float(args.Vcut) / float(I1)
    dV_dtheta = (C / surface.R)  # (Nθ,1)
    f_theta_np = _cut_phase_theta(
        theta=np.asarray(surface.theta), R0=args.R0, a=args.a, theta_cut=args.theta_cut
    )
    V_vis = float(args.Vcut) * f_theta_np
    V_vis = np.asarray(V_vis)[:, None] * np.ones((1, args.n_phi), dtype=float)

    K = (-float(args.sigma_s)) * (dV_dtheta / (surface.a * surface.a))[..., None] * surface.r_theta

    e_theta = surface.r_theta / surface.a
    e_phi = surface.r_phi / jnp.sqrt(surface.G)[..., None]
    Kmag = jnp.linalg.norm(K, axis=-1)
    Ktheta = jnp.sum(K * e_theta, axis=-1)
    Kphi = jnp.sum(K * e_phi, axis=-1)

    def _rms(x):
        return float(jnp.sqrt(jnp.mean(x * x)))

    print("Inboard-cut voltage drive (axisymmetric)")
    print(f"  surface: R0={args.R0} a={args.a} n_theta={args.n_theta} n_phi={args.n_phi}")
    print(f"  cut:     theta_cut={float(args.theta_cut)% (2*np.pi):.3f}  Vcut={args.Vcut:+.3e}  sigma_s={args.sigma_s:+.3e}")
    print(f"  ∮ dθ/R = {float(I1):.6e}  => C={C:+.6e} (so ∂θV=C/R)")
    # Clarify "source/sink" interpretation of the cut.
    # By construction, f_theta is ~0 just *after* the cut index and ~1 just *before* it,
    # so V_vis jumps from ~Vcut to 0 at theta_cut.
    theta_np = np.asarray(surface.theta)
    dtheta = float(2 * np.pi / theta_np.size)
    cut = float(args.theta_cut) % (2 * np.pi)
    diff = _wrap_angle_np(theta_np - cut)
    k0 = int(np.argmin(np.abs(diff)))
    k_minus = (k0 - 1) % theta_np.size
    k_plus = k0
    V_minus = float(args.Vcut) * float(f_theta_np[k_minus])
    V_plus = float(args.Vcut) * float(f_theta_np[k_plus])
    dV = V_minus - V_plus
    print("  cut terminals (interpretation):")
    print(
        "    just before cut (θ≈{:.3f}-Δθ): V≈{:+.3e}".format(
            float(theta_np[k_minus]), V_minus
        )
    )
    print(
        "    just after  cut (θ≈{:.3f}+0): V≈{:+.3e}".format(
            float(theta_np[k_plus]), V_plus
        )
    )
    print(
        "    ΔV = V_before - V_after ≈ {:+.3e} (should be close to Vcut; grid Δθ={:.3e})".format(
            dV, dtheta
        )
    )
    print("Surface-current summary (from K = -σ_s ∇_s V)")
    print(f"  Ktheta_rms={_rms(Ktheta):.3e} A/m   Kphi_rms={_rms(Kphi):.3e} A/m   |K|_rms={_rms(Kmag):.3e} A/m")

    # Evaluate Biot–Savart field on the midplane (phi=0, z=0), inside the torus.
    rho = float(args.rho_eval) * float(args.a)
    R = np.linspace(args.R0 - rho, args.R0 + rho, args.n_eval, dtype=float)
    eval_pts = jnp.stack(
        [
            jnp.asarray(R, dtype=jnp.float64),
            jnp.zeros((args.n_eval,), dtype=jnp.float64),
            jnp.zeros((args.n_eval,), dtype=jnp.float64),
        ],
        axis=-1,
    )

    B_bs = jax.jit(lambda x: biot_savart_surface(surface, K, x, eps=float(args.eps)))(eval_pts)
    B_bs.block_until_ready()

    # At phi=0, e_phi is +y. Use By as Bphi.
    Bphi_bs = np.asarray(B_bs[:, 1], dtype=float)
    idx0 = int(np.argmin(np.abs(R - args.R0)))
    B0_fit = float(Bphi_bs[idx0])
    Bphi_ideal = (B0_fit * args.R0 / R).astype(float)

    rel = float(np.max(np.abs(Bphi_bs - Bphi_ideal) / (np.abs(Bphi_ideal) + 1e-30)))
    print("Toroidal-field comparison (midplane, phi=0)")
    print(f"  fit: B0_fit = Bphi(R=R0) = {B0_fit:+.4e} T")
    print(f"  max_rel_err(Bphi vs B0_fit*R0/R) = {rel:.3e}")

    # Optional: trace an ideal-toroidal field line and validate Biot–Savart along it.
    pts = None
    Bphi_bs_tr = None
    Bphi_an_tr = None
    if args.trace:
        r0 = jnp.array([args.R0 + 0.5 * args.a, 0.0, 0.0], dtype=jnp.float64)
        B_an = lambda xyz: ideal_toroidal_field(xyz, B0=B0_fit, R0=args.R0)

        print("Tracing ideal-toroidal field line (analytic) ...")
        pts = jax.jit(
            lambda x: trace_field_line(
                B_an, x, step_size=float(args.trace_ds), n_steps=int(args.trace_steps), normalize=True
            )
        )(r0)

        idx = jnp.linspace(0, pts.shape[0] - 1, 64, dtype=jnp.int32)
        sample = pts[idx]
        B_bs_tr = biot_savart_surface(surface, K, sample, eps=float(args.eps))
        B_an_tr = B_an(sample)

        R_tr, phi_tr, _ = cylindrical_coords(sample)
        _, ephi_tr, _ = cylindrical_unit_vectors(phi_tr)
        Bphi_bs_tr = np.asarray(jnp.sum(B_bs_tr * ephi_tr, axis=-1))
        Bphi_an_tr = np.asarray(jnp.sum(B_an_tr * ephi_tr, axis=-1))

        abs_err = float(np.max(np.abs(Bphi_bs_tr - Bphi_an_tr)))
        rel_err = float(np.max(np.abs(Bphi_bs_tr - Bphi_an_tr) / (np.abs(Bphi_an_tr) + 1e-30)))
        print("Fieldline sample check (Biot–Savart vs ideal-toroidal along traced line)")
        print(f"  max_abs_err(Bphi)={abs_err:.3e} T  max_rel_err(Bphi)={rel_err:.3e}")

    if not args.no_paraview:
        outdir_p = ensure_dir(args.outdir)
        pv_dir = ensure_dir(outdir_p / "paraview")

        surf_pd = {
            "V": np.asarray(V_vis).reshape(-1),
            "K": np.asarray(K).reshape(-1, 3),
            "Ktheta": np.asarray(Ktheta).reshape(-1),
            "Kphi": np.asarray(Kphi).reshape(-1),
            "|K|": np.asarray(Kmag).reshape(-1),
        }
        surf_vtu = write_vtu(pv_dir / "winding_surface.vtu", torus_surface_to_vtu(surface=surface, point_data=surf_pd))

        eval_pd = {
            "R": np.asarray(R, dtype=float),
            "B": np.asarray(B_bs, dtype=float),
            "Bphi": np.asarray(Bphi_bs, dtype=float),
            "Bphi_ideal": np.asarray(Bphi_ideal, dtype=float),
        }
        eval_vtu = write_vtu(
            pv_dir / "eval_points.vtu",
            point_cloud_to_vtu(points=np.asarray(eval_pts, dtype=float), point_data=eval_pd),
        )

        blocks: dict[str, str] = {
            "winding_surface": surf_vtu.name,
            "eval_points": eval_vtu.name,
        }
        if pts is not None:
            traj_vtu = write_vtu(
                pv_dir / "ideal_fieldline.vtu",
                fieldlines_to_vtu(traj=np.asarray(pts, dtype=float)[None, :, :]),
            )
            blocks["ideal_fieldline"] = traj_vtu.name

        scene = write_vtm(pv_dir / "scene.vtm", blocks)
        print(f"ParaView scene: {scene}")

    if args.no_plots:
        return

    outdir = ensure_dir(args.outdir)

    # Show the cut location as an overlay ring of points (purely for visualization).
    phi_cut = np.linspace(0.0, 2 * np.pi, 80, endpoint=False)
    theta_cut = (float(args.theta_cut) % (2 * np.pi)) * np.ones_like(phi_cut)
    overlay_cut = (phi_cut, theta_cut)

    plot_surface_map(
        phi=np.asarray(surface.phi),
        theta=np.asarray(surface.theta),
        data=V_vis,
        title="Multi-valued cut potential (visualization only): jump at the cut",
        cbar_label=r"$V$ [arb]",
        overlay_points=overlay_cut,
        path=outdir / "surface" / "V_cut_visualization.png",
        cmap="coolwarm",
    )
    plot_surface_map(
        phi=np.asarray(surface.phi),
        theta=np.asarray(surface.theta),
        data=np.asarray(Ktheta),
        title="Poloidal current component $K_\\theta$ (cut-driven, s=0)",
        cbar_label=r"$K_\theta$ [A/m]",
        overlay_points=overlay_cut,
        path=outdir / "surface" / "Ktheta.png",
        cmap="coolwarm",
    )
    plot_surface_map(
        phi=np.asarray(surface.phi),
        theta=np.asarray(surface.theta),
        data=np.asarray(Kmag),
        title="Current magnitude $|K|$ (cut-driven, s=0)",
        cbar_label=r"$|K|$ [A/m]",
        overlay_points=overlay_cut,
        path=outdir / "surface" / "Kmag.png",
    )

    # Bphi vs R on the midplane.
    fig, ax = tplot.plt.subplots(constrained_layout=True)
    ax.plot(R, Bphi_bs, color="tab:blue", label="Biot–Savart (cut-driven K)")
    ax.plot(R, Bphi_ideal, "k--", label=r"fit: $B_\phi=B_0 R_0/R$")
    ax.set_xlabel("R [m] (midplane, Z=0, φ=0)")
    ax.set_ylabel(r"$B_\phi$ [T]")
    ax.set_title("Toroidal field vs 1/R scaling (midplane)")
    ax.legend(loc="best")
    savefig(fig, outdir / "field" / "Bphi_vs_R.png", dpi=args.dpi)

    # 3D geometry: torus + eval points, plus optional traced curve.
    plot_3d_torus(
        torus_xyz=np.asarray(surface.r),
        electrodes_xyz=None,
        curve_xyz=None if pts is None else np.asarray(pts),
        eval_xyz=np.asarray(eval_pts),
        stride=args.plot_stride,
        title="Torus + eval points" + (" + ideal toroidal fieldline" if pts is not None else ""),
        path=outdir / "geometry" / "torus_and_eval.png",
    )

    if pts is not None:
        plot_fieldline(
            pts=np.asarray(pts),
            title="Ideal toroidal field line (analytic, B0 fit from cut-driven field)",
            R0=args.R0,
            path3d=outdir / "traj" / "fieldline_3d.png",
            path_rz=outdir / "traj" / "fieldline_RZ.png",
        )

        fig, ax = tplot.plt.subplots(constrained_layout=True)
        sidx = np.arange(Bphi_bs_tr.size)
        ax.plot(sidx, Bphi_an_tr, "k--", label="ideal (analytic)")
        ax.plot(sidx, Bphi_bs_tr, color="tab:blue", label="Biot–Savart (cut-driven K)")
        ax.set_xlabel("Sample index along traced line")
        ax.set_ylabel(r"$B_\phi$ [T]")
        ax.set_title(r"$B_\phi$ along traced line: Biot–Savart vs ideal")
        ax.legend(loc="best")
        savefig(fig, outdir / "field" / "Bphi_along_trace.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
