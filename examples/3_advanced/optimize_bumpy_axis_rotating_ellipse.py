#!/usr/bin/env python3
"""Example: target a rotating-ellipse, bumpy magnetic axis and optimize electrodes.

This is a deliberately "toy near-axis" target:
  - A bumpy magnetic axis is prescribed in cylindrical coordinates:
        R(φ) = R_axis0 + dr * cos(nfp φ)
        Z(φ) = dz * sin(nfp φ)
  - A local Frenet-like frame (t,n,b) is built numerically along the axis.
  - The target field is a mostly-tangent field with a rotating transverse component:
        B_target = B0 t + B1 (cos(α) n + sin(α) b),  α = nfp φ + α0

Run:
  - `python examples/3_advanced/optimize_bumpy_axis_rotating_ellipse.py`
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
import time

import jax
import jax.numpy as jnp
import numpy as np

from torus_solver import make_torus_surface
from torus_solver.optimize import SourceParams, forward_B, optimize_sources, surface_solution
from torus_solver.paraview import fieldlines_to_vtu, point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu
from torus_solver.plotting import (
    ensure_dir,
    plot_3d_torus,
    plot_fieldlines_3d,
    plot_loss_history,
    plot_surface_map,
    savefig,
    set_plot_style,
)
import torus_solver.plotting as tplot


def _normalize(v: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


def bumpy_axis_points(
    phi: jnp.ndarray, *, R_axis0: float, dr: float, dz: float, nfp: int
) -> jnp.ndarray:
    R = R_axis0 + dr * jnp.cos(nfp * phi)
    Z = dz * jnp.sin(nfp * phi)
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    return jnp.stack([x, y, Z], axis=-1)


def discrete_frenet_frame(points: jnp.ndarray, *, eps: float = 1e-12):
    """Approximate (t,n,b) from a closed curve sampled uniformly in φ."""
    dp = 0.5 * (jnp.roll(points, -1, axis=0) - jnp.roll(points, 1, axis=0))
    t = _normalize(dp, eps=eps)

    dt = 0.5 * (jnp.roll(t, -1, axis=0) - jnp.roll(t, 1, axis=0))
    dt_perp = dt - jnp.sum(dt * t, axis=-1, keepdims=True) * t
    n = _normalize(dt_perp, eps=eps)
    b = _normalize(jnp.cross(t, n), eps=eps)
    return t, n, b


def make_target(
    *,
    R0: float,
    a: float,
    n_axis: int,
    nfp: int,
    dr_axis: float,
    dz_axis: float,
    B0: float,
    B1: float,
    alpha0: float,
    ellipse_a: float,
    ellipse_b: float,
    n_psi: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    phi = jnp.linspace(0.0, 2 * jnp.pi, n_axis, endpoint=False, dtype=jnp.float64)
    axis = bumpy_axis_points(phi, R_axis0=R0, dr=dr_axis, dz=dz_axis, nfp=nfp)

    t, n, b = discrete_frenet_frame(axis)
    alpha = nfp * phi + alpha0
    n_rot = jnp.cos(alpha)[:, None] * n + jnp.sin(alpha)[:, None] * b
    b_rot = -jnp.sin(alpha)[:, None] * n + jnp.cos(alpha)[:, None] * b

    # Evaluation points: a small rotating ellipse around the axis.
    psi = jnp.linspace(0.0, 2 * jnp.pi, n_psi, endpoint=False, dtype=jnp.float64)
    c = jnp.cos(psi)[None, :, None]
    s = jnp.sin(psi)[None, :, None]
    pts = axis[:, None, :] + ellipse_a * c * n_rot[:, None, :] + ellipse_b * s * b_rot[:, None, :]
    eval_points = pts.reshape((-1, 3))

    # Target field is specified on those points using the axis frame at each φ.
    B_axis = B0 * t + B1 * n_rot
    B_target = jnp.broadcast_to(B_axis[:, None, :], pts.shape).reshape((-1, 3))

    # Debug info:
    print("Target construction")
    print(f"  axis: n_axis={n_axis} nfp={nfp} dr_axis={dr_axis} dz_axis={dz_axis}")
    print(f"  ellipse: a={ellipse_a} b={ellipse_b} n_psi={n_psi} -> n_eval={eval_points.shape[0]}")
    print(f"  |B_target| mean={float(jnp.mean(jnp.linalg.norm(B_target, axis=-1))):.3e}")

    # Sanity check orthonormality at a few points.
    dot_tn = jnp.mean(jnp.sum(t * n, axis=-1))
    dot_tb = jnp.mean(jnp.sum(t * b, axis=-1))
    dot_nb = jnp.mean(jnp.sum(n * b, axis=-1))
    print(f"  frame dots mean: t·n={float(dot_tn):+.2e} t·b={float(dot_tb):+.2e} n·b={float(dot_nb):+.2e}")

    return phi, axis, pts, B_target


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0, help="Winding-surface major radius (m)")
    p.add_argument("--a", type=float, default=1.0, help="Winding-surface minor radius (m)")
    p.add_argument("--n-theta", type=int, default=40)
    p.add_argument("--n-phi", type=int, default=40)
    p.add_argument("--n-sources", type=int, default=12)
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-2)

    p.add_argument("--n-axis", type=int, default=16)
    p.add_argument("--n-psi", type=int, default=6)
    p.add_argument("--nfp", type=int, default=2)
    p.add_argument("--dr-axis", type=float, default=0.15, help="Axis radial bump amplitude (m)")
    p.add_argument("--dz-axis", type=float, default=0.10, help="Axis vertical bump amplitude (m)")
    p.add_argument("--ellipse-a", type=float, default=0.12, help="Rotating ellipse semi-axis a (m)")
    p.add_argument("--ellipse-b", type=float, default=0.06, help="Rotating ellipse semi-axis b (m)")
    p.add_argument("--alpha0", type=float, default=0.0)

    p.add_argument("--B0", type=float, default=1e-4, help="Target field scale (T)")
    p.add_argument("--B1", type=float, default=2.5e-5, help="Transverse component amplitude (T)")

    p.add_argument("--sigma-theta", type=float, default=0.25)
    p.add_argument("--sigma-phi", type=float, default=0.25)
    p.add_argument("--outdir", type=str, default="figures/optimize_bumpy_axis_rotating_ellipse")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true", help="Disable ParaView (.vtu/.vtm) outputs")
    p.add_argument("--plot-stride", type=int, default=3)
    p.add_argument("--no-fieldlines", action="store_true", help="Disable final 3D field-line plot (faster)")
    p.add_argument("--n-fieldlines", type=int, default=12)
    p.add_argument("--fieldline-steps", type=int, default=500)
    p.add_argument("--fieldline-ds", type=float, default=0.03)
    p.add_argument("--biot-savart-eps", type=float, default=1e-8)
    p.add_argument(
        "--Bext0",
        type=float,
        default=0.0,
        help="Optional external ideal toroidal background field at R=R0 [T] (0 disables).",
    )

    args = p.parse_args()

    outdir = None
    if (not args.no_plots) or (not args.no_paraview):
        outdir = ensure_dir(args.outdir)
        if not args.no_plots:
            set_plot_style()
            ensure_dir(outdir / "surface")
            ensure_dir(outdir / "target")
            ensure_dir(outdir / "optim")
            ensure_dir(outdir / "geometry")
            ensure_dir(outdir / "fieldlines")
            print(f"Saving figures to: {outdir}")
        if not args.no_paraview:
            ensure_dir(outdir / "paraview")
            print(f"Saving ParaView outputs to: {outdir / 'paraview'}")

    print("Building winding surface (circular torus)")
    print(f"  R0={args.R0} a={args.a} n_theta={args.n_theta} n_phi={args.n_phi}")
    surface = make_torus_surface(R0=args.R0, a=args.a, n_theta=args.n_theta, n_phi=args.n_phi)

    phi_axis, axis_xyz, pts_ellipse, B_target = make_target(
        R0=args.R0,
        a=args.a,
        n_axis=args.n_axis,
        nfp=args.nfp,
        dr_axis=args.dr_axis,
        dz_axis=args.dz_axis,
        B0=args.B0,
        B1=args.B1,
        alpha0=args.alpha0,
        ellipse_a=args.ellipse_a,
        ellipse_b=args.ellipse_b,
        n_psi=args.n_psi,
    )
    eval_points = pts_ellipse.reshape((-1, 3))

    key = jax.random.key(0)
    theta_src = 2 * jnp.pi * jax.random.uniform(key, (args.n_sources,))
    phi_src = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(key, 1), (args.n_sources,))
    currents_raw = 1e3 * jax.random.normal(jax.random.fold_in(key, 2), (args.n_sources,))
    init = SourceParams(theta_src=theta_src, phi_src=phi_src, currents_raw=currents_raw)

    # Evaluate initial loss.
    B_init = forward_B(
        surface,
        init,
        eval_points=eval_points,
        sigma_theta=args.sigma_theta,
        sigma_phi=args.sigma_phi,
    )
    rms0 = jnp.sqrt(jnp.mean(jnp.sum((B_init - B_target) ** 2, axis=-1)))
    print(f"Initial RMS(B-Btgt) = {float(rms0):.6e} T")

    def cb(step: int, metrics: dict[str, float]) -> None:
        if step % 20 == 0 or step == args.n_steps - 1:
            print(
                "step={:4d} loss={:.6e} loss_B={:.6e} loss_reg={:.3e} I_rms={:.3e} A".format(
                    step, metrics["loss"], metrics["loss_B"], metrics["loss_reg"], metrics["currents_rms"]
                )
            )

    t0 = time.perf_counter()
    best, history = optimize_sources(
        surface,
        init=init,
        eval_points=eval_points,
        B_target=B_target,
        B_scale=args.B0,
        sigma_theta=args.sigma_theta,
        sigma_phi=args.sigma_phi,
        n_steps=args.n_steps,
        lr=args.lr,
        reg_currents=1e-12,
        callback=cb,
        return_history=True,
    )
    t1 = time.perf_counter()

    B_final = forward_B(
        surface,
        best,
        eval_points=eval_points,
        sigma_theta=args.sigma_theta,
        sigma_phi=args.sigma_phi,
    )
    rms = jnp.sqrt(jnp.mean(jnp.sum((B_final - B_target) ** 2, axis=-1)))

    currents = best.currents_raw - jnp.mean(best.currents_raw)
    print("Optimization done")
    print(f"  wall_time_s={t1 - t0:.2f}")
    print(f"  final_rms_T={float(rms):.6e}")
    print(f"  currents: mean={float(jnp.mean(currents)):+.3e} A  rms={float(jnp.sqrt(jnp.mean(currents**2))):.3e} A")
    print(f"  currents: min={float(jnp.min(currents)):+.3e} A  max={float(jnp.max(currents)):+.3e} A")
    if float(args.Bext0) != 0.0:
        print(f"  fieldlines: include external ideal toroidal field Bext0={args.Bext0} T at R=R0")

    need_surface_fields = (not args.no_plots) or (not args.no_paraview)
    if need_surface_fields:
        currents0, s0, V0, K0 = surface_solution(surface, init, sigma_theta=args.sigma_theta, sigma_phi=args.sigma_phi)
        currents1, s1, V1, K1 = surface_solution(surface, best, sigma_theta=args.sigma_theta, sigma_phi=args.sigma_phi)

        e_theta = surface.r_theta / surface.a
        e_phi_s = surface.r_phi / jnp.sqrt(surface.G)[..., None]
        Ktheta0 = jnp.sum(K0 * e_theta, axis=-1)
        Kphi0 = jnp.sum(K0 * e_phi_s, axis=-1)
        Ktheta1 = jnp.sum(K1 * e_theta, axis=-1)
        Kphi1 = jnp.sum(K1 * e_phi_s, axis=-1)
        Kmag0 = jnp.sqrt(Ktheta0 * Ktheta0 + Kphi0 * Kphi0)
        Kmag1 = jnp.sqrt(Ktheta1 * Ktheta1 + Kphi1 * Kphi1)

    traj_np = None
    if need_surface_fields and (not args.no_fieldlines):
        print("Tracing field lines for final solution (Biot–Savart)...")
        from torus_solver.biot_savart import biot_savart_surface
        from torus_solver.fieldline import trace_field_lines_batch
        from torus_solver.fields import ideal_toroidal_field

        theta_seed = jnp.linspace(0.0, 2 * jnp.pi, int(args.n_fieldlines), endpoint=False)
        rho = 0.5 * surface.a
        R_seed = surface.R0 + rho * jnp.cos(theta_seed)
        Z_seed = rho * jnp.sin(theta_seed)
        seeds = jnp.stack([R_seed, jnp.zeros_like(R_seed), Z_seed], axis=-1)

        def B_fn(xyz: jnp.ndarray) -> jnp.ndarray:
            B = biot_savart_surface(surface, K1, xyz, eps=float(args.biot_savart_eps))
            if float(args.Bext0) != 0.0:
                B = B + ideal_toroidal_field(xyz, B0=float(args.Bext0), R0=float(surface.R0))
            return B

        traj = jax.jit(
            lambda x0: trace_field_lines_batch(
                B_fn,
                x0,
                step_size=float(args.fieldline_ds),
                n_steps=int(args.fieldline_steps),
                normalize=True,
            )
        )(seeds)
        traj.block_until_ready()
        traj_np = np.asarray(traj)

    if not args.no_paraview:
        assert outdir is not None
        pv_dir = ensure_dir(outdir / "paraview")

        surf_init = write_vtu(
            pv_dir / "winding_surface_init.vtu",
            torus_surface_to_vtu(
                surface=surface,
                point_data={
                    "V": np.asarray(V0).reshape(-1),
                    "s": np.asarray(s0).reshape(-1),
                    "K": np.asarray(K0).reshape(-1, 3),
                    "Ktheta": np.asarray(Ktheta0).reshape(-1),
                    "Kphi": np.asarray(Kphi0).reshape(-1),
                    "|K|": np.asarray(Kmag0).reshape(-1),
                },
            ),
        )
        surf_final = write_vtu(
            pv_dir / "winding_surface_final.vtu",
            torus_surface_to_vtu(
                surface=surface,
                point_data={
                    "V": np.asarray(V1).reshape(-1),
                    "s": np.asarray(s1).reshape(-1),
                    "K": np.asarray(K1).reshape(-1, 3),
                    "Ktheta": np.asarray(Ktheta1).reshape(-1),
                    "Kphi": np.asarray(Kphi1).reshape(-1),
                    "|K|": np.asarray(Kmag1).reshape(-1),
                },
            ),
        )

        def torus_xyz(theta, phi):
            R = args.R0 + args.a * np.cos(theta)
            return np.stack([R * np.cos(phi), R * np.sin(phi), args.a * np.sin(theta)], axis=-1)

        el0_xyz = torus_xyz(np.asarray(init.theta_src), np.asarray(init.phi_src))
        el1_xyz = torus_xyz(np.asarray(best.theta_src), np.asarray(best.phi_src))
        el_init = write_vtu(
            pv_dir / "electrodes_init.vtu",
            point_cloud_to_vtu(
                points=el0_xyz,
                point_data={
                    "I_A": np.asarray(currents0, dtype=float),
                    "I_raw": np.asarray(init.currents_raw, dtype=float),
                    "sign_I": np.sign(np.asarray(currents0, dtype=float)),
                },
            ),
        )
        el_final = write_vtu(
            pv_dir / "electrodes_final.vtu",
            point_cloud_to_vtu(
                points=el1_xyz,
                point_data={
                    "I_A": np.asarray(currents1, dtype=float),
                    "I_raw": np.asarray(best.currents_raw, dtype=float),
                    "sign_I": np.sign(np.asarray(currents1, dtype=float)),
                },
            ),
        )

        tgt_vtu = write_vtu(
            pv_dir / "target_points.vtu",
            point_cloud_to_vtu(
                points=np.asarray(eval_points, dtype=float),
                point_data={
                    "B_target": np.asarray(B_target, dtype=float),
                    "B_init": np.asarray(B_init, dtype=float),
                    "B_final": np.asarray(B_final, dtype=float),
                },
            ),
        )
        axis_vtu = write_vtu(
            pv_dir / "axis.vtu",
            fieldlines_to_vtu(traj=np.asarray(axis_xyz, dtype=float)[None, :, :]),
        )

        blocks: dict[str, str] = {
            "winding_surface_init": surf_init.name,
            "winding_surface_final": surf_final.name,
            "electrodes_init": el_init.name,
            "electrodes_final": el_final.name,
            "target_points": tgt_vtu.name,
            "axis": axis_vtu.name,
        }
        if traj_np is not None:
            # `trace_field_lines_batch` returns (n_steps+1, n_lines, 3), while ParaView
            # polylines are easiest to write as (n_lines, n_points, 3).
            traj_pv = np.transpose(traj_np, (1, 0, 2))
            fl_vtu = write_vtu(pv_dir / "fieldlines_final.vtu", fieldlines_to_vtu(traj=traj_pv))
            blocks["fieldlines_final"] = fl_vtu.name

        scene = write_vtm(pv_dir / "scene.vtm", blocks)
        print(f"ParaView scene: {scene}")

    if args.no_plots:
        return

    assert outdir is not None

    if not args.no_plots:
        # Surface maps (final).
        phi_grid = np.asarray(surface.phi)
        theta_grid = np.asarray(surface.theta)

        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(V0),
            title="Surface potential V (initial)",
            cbar_label="V [arb.]",
            path=outdir / "surface" / "V_init.png",
            overlay_points=(np.asarray(init.phi_src), np.asarray(init.theta_src)),
        )
        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(V1),
            title="Surface potential V (final)",
            cbar_label="V [arb.]",
            path=outdir / "surface" / "V_final.png",
            overlay_points=(np.asarray(best.phi_src), np.asarray(best.theta_src)),
        )
        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(s0),
            title="Source density s (initial)",
            cbar_label=r"$s$ [A/m$^2$]",
            path=outdir / "surface" / "s_init.png",
            cmap="coolwarm",
            overlay_points=(np.asarray(init.phi_src), np.asarray(init.theta_src)),
        )
        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(s1),
            title="Source density s (final)",
            cbar_label=r"$s$ [A/m$^2$]",
            path=outdir / "surface" / "s_final.png",
            cmap="coolwarm",
            overlay_points=(np.asarray(best.phi_src), np.asarray(best.theta_src)),
        )

        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(Kmag0),
            title="|K| (initial)",
            cbar_label=r"$|K|$ [A/m]",
            path=outdir / "surface" / "Kmag_init.png",
        )
        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(Kmag1),
            title="|K| (final)",
            cbar_label=r"$|K|$ [A/m]",
            path=outdir / "surface" / "Kmag_final.png",
        )

        # Target geometry plots.
        R_axis = np.sqrt(np.asarray(axis_xyz[:, 0]) ** 2 + np.asarray(axis_xyz[:, 1]) ** 2)
        Z_axis = np.asarray(axis_xyz[:, 2])
        fig, ax = tplot.plt.subplots(constrained_layout=True)
        ax.plot(np.asarray(phi_axis), R_axis, label="R(φ)")
        ax.plot(np.asarray(phi_axis), Z_axis, label="Z(φ)")
        ax.set_xlabel(r"$\phi$ [rad]")
        ax.set_ylabel("[m]")
        ax.set_title("Prescribed bumpy axis in cylindrical coordinates")
        ax.legend()
        savefig(fig, outdir / "target" / "axis_RZ_vs_phi.png", dpi=args.dpi)

        # Error before/after on ellipse points.
        err0 = np.linalg.norm(np.asarray(B_init - B_target), axis=-1)
        err1 = np.linalg.norm(np.asarray(B_final - B_target), axis=-1)
        fig, ax = tplot.plt.subplots(constrained_layout=True)
        ax.hist(err0, bins=30, alpha=0.55, label="init")
        ax.hist(err1, bins=30, alpha=0.55, label="final")
        ax.set_xlabel(r"$|B-B_{target}|$ [T]")
        ax.set_ylabel("Count")
        ax.set_title("Pointwise field error distribution on target ellipse")
        ax.legend()
        savefig(fig, outdir / "target" / "error_hist.png", dpi=args.dpi)

        # Per-axis-section RMS (average over ellipse angle ψ).
        n_axis = args.n_axis
        n_psi = args.n_psi
        err0_grid = err0.reshape((n_axis, n_psi))
        err1_grid = err1.reshape((n_axis, n_psi))
        rms0_phi = np.sqrt(np.mean(err0_grid**2, axis=1))
        rms1_phi = np.sqrt(np.mean(err1_grid**2, axis=1))

        fig, ax = tplot.plt.subplots(constrained_layout=True)
        ax.plot(np.asarray(phi_axis), rms0_phi, label="init")
        ax.plot(np.asarray(phi_axis), rms1_phi, label="final")
        ax.set_xlabel(r"$\phi$ [rad]")
        ax.set_ylabel("RMS error on ellipse [T]")
        ax.set_title("Field error vs axis toroidal angle (ellipse-averaged)")
        ax.legend()
        savefig(fig, outdir / "target" / "error_vs_phi.png", dpi=args.dpi)

        # Optimization history.
        steps = list(range(len(history["loss"])))
        plot_loss_history(
            steps=steps,
            metrics={
                "loss": history["loss"],
                "loss_B": history["loss_B"],
                "loss_reg": history["loss_reg"],
            },
            title="Optimization history (log scale)",
            path=outdir / "optim" / "loss_history.png",
        )
        plot_loss_history(
            steps=steps,
            metrics={"I_rms [A]": history["currents_rms"]},
            title="Electrode currents RMS",
            path=outdir / "optim" / "currents_rms.png",
        )

        # 3D geometry: torus + axis + eval points + electrodes.
        def torus_xyz(theta, phi):
            R = args.R0 + args.a * np.cos(theta)
            return np.stack([R * np.cos(phi), R * np.sin(phi), args.a * np.sin(theta)], axis=-1)

        electrodes0 = torus_xyz(np.asarray(init.theta_src), np.asarray(init.phi_src))
        electrodes1 = torus_xyz(np.asarray(best.theta_src), np.asarray(best.phi_src))
        plot_3d_torus(
            torus_xyz=np.asarray(surface.r),
            electrodes_xyz=electrodes0,
            curve_xyz=np.asarray(axis_xyz),
            eval_xyz=np.asarray(eval_points),
            stride=args.plot_stride,
            title="Torus + bumpy axis + evaluation points + initial electrodes",
            path=outdir / "geometry" / "geometry_init.png",
        )
        plot_3d_torus(
            torus_xyz=np.asarray(surface.r),
            electrodes_xyz=electrodes1,
            curve_xyz=np.asarray(axis_xyz),
            eval_xyz=np.asarray(eval_points),
            stride=args.plot_stride,
            title="Torus + bumpy axis + evaluation points + final electrodes",
            path=outdir / "geometry" / "geometry_final.png",
        )

        if traj_np is not None:
            plot_fieldlines_3d(
                torus_xyz=np.asarray(surface.r),
                traj=traj_np,
                stride=args.plot_stride,
                line_stride=1,
                title="Final field lines (Biot–Savart from optimized surface currents)",
                path=outdir / "fieldlines" / "fieldlines_final.png",
            )

        # Currents bar plot.
        fig, ax = tplot.plt.subplots(constrained_layout=True)
        idx = np.arange(args.n_sources)
        w = 0.38
        ax.bar(idx - w / 2, np.asarray(currents0), width=w, label="init")
        ax.bar(idx + w / 2, np.asarray(currents1), width=w, label="final")
        ax.set_xlabel("Electrode index")
        ax.set_ylabel("Injected current [A]")
        ax.set_title("Electrode currents (projected to net zero)")
        ax.legend()
        savefig(fig, outdir / "optim" / "currents_bar.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
