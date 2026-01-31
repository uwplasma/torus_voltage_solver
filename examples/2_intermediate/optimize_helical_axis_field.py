#!/usr/bin/env python3
"""Example: optimize source/sink electrodes to reproduce a helical field on-axis.

Run:
  - `python examples/2_intermediate/optimize_helical_axis_field.py`
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
from torus_solver.optimize import (
    SourceParams,
    enforce_net_zero,
    forward_B,
    make_helical_axis_points,
    optimize_sources,
    optimize_sources_lbfgs,
    surface_solution,
)
from torus_solver.paraview import fieldlines_to_vtu, point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu
from torus_solver.plotting import (
    ensure_dir,
    plot_3d_torus,
    plot_axis_field_comparison,
    plot_fieldlines_3d,
    plot_loss_history,
    plot_surface_map,
    savefig,
    set_plot_style,
)
import torus_solver.plotting as tplot


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0)
    p.add_argument("--a", type=float, default=1.0)
    p.add_argument("--n-theta", type=int, default=48)
    p.add_argument("--n-phi", type=int, default=48)
    p.add_argument("--n-points", type=int, default=32)
    p.add_argument("--n-sources", type=int, default=12)
    p.add_argument("--nfp", type=int, default=2)
    p.add_argument("--B0", type=float, default=1e-4)
    p.add_argument("--B1", type=float, default=None)
    p.add_argument("--sigma-theta", type=float, default=0.25)
    p.add_argument("--sigma-phi", type=float, default=0.25)
    p.add_argument("--optimizer", type=str, default="adam", choices=["adam", "lbfgs"])
    p.add_argument("--lbfgs-tol", type=float, default=1e-9)
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--outdir", type=str, default="figures/optimize_helical_axis_field")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true", help="Disable ParaView (.vtu/.vtm) outputs")
    p.add_argument("--plot-stride", type=int, default=3, help="Decimation stride for 3D torus wireframe")
    p.add_argument("--no-fieldlines", action="store_true", help="Disable final 3D field-line plot (faster)")
    p.add_argument("--n-fieldlines", type=int, default=12)
    p.add_argument("--fieldline-steps", type=int, default=500)
    p.add_argument("--fieldline-ds", type=float, default=0.03)
    p.add_argument("--biot-savart-eps", type=float, default=1e-8)
    args = p.parse_args()

    if args.B1 is None:
        args.B1 = 0.25 * args.B0

    outdir = None
    if (not args.no_plots) or (not args.no_paraview):
        outdir = ensure_dir(args.outdir)
        if not args.no_plots:
            set_plot_style()
            ensure_dir(outdir / "surface")
            ensure_dir(outdir / "axis")
            ensure_dir(outdir / "optim")
            ensure_dir(outdir / "geometry")
            ensure_dir(outdir / "fieldlines")
            print(f"Saving figures to: {outdir}")
        if not args.no_paraview:
            ensure_dir(outdir / "paraview")
            print(f"Saving ParaView outputs to: {outdir / 'paraview'}")

    surface = make_torus_surface(R0=args.R0, a=args.a, n_theta=args.n_theta, n_phi=args.n_phi)

    # Target points on the magnetic axis (major-radius circle).
    phi, points, e_r, e_phi, e_z = make_helical_axis_points(
        R_axis=surface.R0, n_points=args.n_points
    )

    # A toy stellarator-like target: toroidal + rotating (r,z) component.
    B_target = args.B0 * e_phi + args.B1 * (
        jnp.cos(args.nfp * phi)[:, None] * e_r + jnp.sin(args.nfp * phi)[:, None] * e_z
    )

    key = jax.random.key(0)
    theta_src = 2 * jnp.pi * jax.random.uniform(key, (args.n_sources,))
    phi_src = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(key, 1), (args.n_sources,))
    currents_raw = 1e3 * jax.random.normal(jax.random.fold_in(key, 2), (args.n_sources,))
    init = SourceParams(theta_src=theta_src, phi_src=phi_src, currents_raw=currents_raw)

    sigma_theta = args.sigma_theta
    sigma_phi = args.sigma_phi

    def cb(step: int, metrics: dict[str, float]) -> None:
        if step % 25 == 0:
            print(
                "step={:4d} loss={:.6e} loss_B={:.6e} loss_reg={:.3e} I_rms={:.3e} A".format(
                    step, metrics["loss"], metrics["loss_B"], metrics["loss_reg"], metrics["currents_rms"]
                )
            )

    print("Helical target optimization")
    print(f"  surface: R0={args.R0} a={args.a} n_theta={args.n_theta} n_phi={args.n_phi}")
    print(f"  target:  nfp={args.nfp} B0={args.B0}T B1={args.B1}T n_points={args.n_points}")
    print(f"  sources: n_sources={args.n_sources} sigma_theta={sigma_theta} sigma_phi={sigma_phi}")
    if args.optimizer == "lbfgs":
        print(f"  optim:   method=lbfgs maxiter={args.n_steps} tol={args.lbfgs_tol}")
    else:
        print(f"  optim:   method=adam n_steps={args.n_steps} lr={args.lr}")

    B_init = forward_B(
        surface,
        init,
        eval_points=points,
        sigma_theta=sigma_theta,
        sigma_phi=sigma_phi,
    )
    rms0 = jnp.sqrt(jnp.mean(jnp.sum((B_init - B_target) ** 2, axis=-1)))
    print(f"initial_rms={float(rms0):.6e} T")

    t0 = time.perf_counter()
    if args.optimizer == "lbfgs":
        best, history = optimize_sources_lbfgs(
            surface,
            init=init,
            eval_points=points,
            B_target=B_target,
            B_scale=args.B0,
            sigma_theta=sigma_theta,
            sigma_phi=sigma_phi,
            maxiter=args.n_steps,
            tol=args.lbfgs_tol,
            reg_currents=1e-12,
            callback=cb,
            return_history=True,
        )
    else:
        best, history = optimize_sources(
            surface,
            init=init,
            eval_points=points,
            B_target=B_target,
            B_scale=args.B0,
            sigma_theta=sigma_theta,
            sigma_phi=sigma_phi,
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
        eval_points=points,
        sigma_theta=sigma_theta,
        sigma_phi=sigma_phi,
    )
    rms = jnp.sqrt(jnp.mean(jnp.sum((B_final - B_target) ** 2, axis=-1)))
    print(f"final_rms={float(rms):.6e} T  wall_time_s={t1 - t0:.2f}")
    print(f"net_current={float(jnp.sum(enforce_net_zero(best.currents_raw))):+.3e} A")

    need_surface_fields = (not args.no_plots) or (not args.no_paraview)
    if need_surface_fields:
        # Surface fields (init/final) for maps + ParaView.
        currents0, s0, V0, K0 = surface_solution(surface, init, sigma_theta=sigma_theta, sigma_phi=sigma_phi)
        currents1, s1, V1, K1 = surface_solution(surface, best, sigma_theta=sigma_theta, sigma_phi=sigma_phi)

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

        theta_seed = jnp.linspace(0.0, 2 * jnp.pi, args.n_fieldlines, endpoint=False)
        rho = 0.5 * surface.a
        R_seed = surface.R0 + rho * jnp.cos(theta_seed)
        Z_seed = rho * jnp.sin(theta_seed)
        seeds = jnp.stack([R_seed, jnp.zeros_like(R_seed), Z_seed], axis=-1)

        def B_fn(xyz: jnp.ndarray) -> jnp.ndarray:
            return biot_savart_surface(surface, K1, xyz, eps=float(args.biot_savart_eps))

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

        # Surface datasets (init/final).
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

        # Electrode point clouds.
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

        # Axis curve (polyline) with target/init/final B vectors.
        axis_vtu = write_vtu(
            pv_dir / "axis.vtu",
            fieldlines_to_vtu(
                traj=np.asarray(points, dtype=float)[None, :, :],
                point_data={
                    "B_target": np.asarray(B_target, dtype=float),
                    "B_init": np.asarray(B_init, dtype=float),
                    "B_final": np.asarray(B_final, dtype=float),
                },
            ),
        )

        blocks: dict[str, str] = {
            "winding_surface_init": surf_init.name,
            "winding_surface_final": surf_final.name,
            "electrodes_init": el_init.name,
            "electrodes_final": el_final.name,
            "axis": axis_vtu.name,
        }
        if traj_np is not None:
            # `trace_field_lines_batch` returns (n_steps+1, n_lines, 3), while ParaView
            # polylines are easiest to write as (n_lines, n_points, 3).
            traj_pv = np.transpose(traj_np, (1, 0, 2))
            fl_vtu = write_vtu(
                pv_dir / "fieldlines_final.vtu",
                fieldlines_to_vtu(traj=traj_pv),
            )
            blocks["fieldlines_final"] = fl_vtu.name

        scene = write_vtm(pv_dir / "scene.vtm", blocks)
        print(f"ParaView scene: {scene}")

    if not args.no_plots:
        assert outdir is not None

        # 2D surface maps (θ vertical, φ horizontal).
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
            data=np.asarray(np.sqrt(Ktheta0**2 + Kphi0**2)),
            title="|K| (initial)",
            cbar_label=r"$|K|$ [A/m]",
            path=outdir / "surface" / "Kmag_init.png",
        )
        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(np.sqrt(Ktheta1**2 + Kphi1**2)),
            title="|K| (final)",
            cbar_label=r"$|K|$ [A/m]",
            path=outdir / "surface" / "Kmag_final.png",
        )
        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(Ktheta1),
            title=r"$K_\theta$ (final)",
            cbar_label=r"$K_\theta$ [A/m]",
            path=outdir / "surface" / "Ktheta_final.png",
            cmap="coolwarm",
        )
        plot_surface_map(
            phi=phi_grid,
            theta=theta_grid,
            data=np.asarray(Kphi1),
            title=r"$K_\phi$ (final)",
            cbar_label=r"$K_\phi$ [A/m]",
            path=outdir / "surface" / "Kphi_final.png",
            cmap="coolwarm",
        )

        # Axis field comparison plots.
        plot_axis_field_comparison(
            phi=np.asarray(phi),
            B_target=np.asarray(B_target),
            B_init=np.asarray(B_init),
            B_final=np.asarray(B_final),
            basis=(np.asarray(e_r), np.asarray(e_phi), np.asarray(e_z)),
            title="On-axis field: target vs init vs final",
            path=outdir / "axis" / "B_components.png",
        )

        # Axis error vs phi.
        err0 = np.linalg.norm(np.asarray(B_init - B_target), axis=-1)
        err1 = np.linalg.norm(np.asarray(B_final - B_target), axis=-1)
        fig, ax = tplot.plt.subplots(constrained_layout=True)
        ax.plot(np.asarray(phi), err0, label="init")
        ax.plot(np.asarray(phi), err1, label="final")
        ax.set_xlabel(r"$\phi$ [rad]")
        ax.set_ylabel(r"$|B-B_{target}|$ [T]")
        ax.set_title("On-axis field error vs toroidal angle")
        ax.legend()
        savefig(fig, outdir / "axis" / "error_vs_phi.png", dpi=args.dpi)

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

        # Geometry 3D.
        def torus_xyz(theta, phi):
            R = args.R0 + args.a * np.cos(theta)
            return np.stack([R * np.cos(phi), R * np.sin(phi), args.a * np.sin(theta)], axis=-1)

        electrodes0 = torus_xyz(np.asarray(init.theta_src), np.asarray(init.phi_src))
        electrodes1 = torus_xyz(np.asarray(best.theta_src), np.asarray(best.phi_src))
        axis_xyz = np.asarray(points)

        plot_3d_torus(
            torus_xyz=np.asarray(surface.r),
            electrodes_xyz=electrodes0,
            curve_xyz=axis_xyz,
            eval_xyz=axis_xyz,
            stride=args.plot_stride,
            title="Torus + initial electrodes + axis",
            path=outdir / "geometry" / "torus_init.png",
        )
        plot_3d_torus(
            torus_xyz=np.asarray(surface.r),
            electrodes_xyz=electrodes1,
            curve_xyz=axis_xyz,
            eval_xyz=axis_xyz,
            stride=args.plot_stride,
            title="Torus + final electrodes + axis",
            path=outdir / "geometry" / "torus_final.png",
        )

        # 3D field lines (final), using Biot–Savart from the optimized surface currents.
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
        I0 = np.asarray(currents0)
        I1 = np.asarray(currents1)
        w = 0.38
        ax.bar(idx - w / 2, I0, width=w, label="init")
        ax.bar(idx + w / 2, I1, width=w, label="final")
        ax.set_xlabel("Electrode index")
        ax.set_ylabel("Injected current [A]")
        ax.set_title("Electrode currents (projected to net zero)")
        ax.legend()
        savefig(fig, outdir / "optim" / "currents_bar.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
