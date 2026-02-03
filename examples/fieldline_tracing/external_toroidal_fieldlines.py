#!/usr/bin/env python3
"""Example: field-line tracing in an *external* ideal toroidal field B ~ 1/R.

This example is intentionally simple and focuses on the "background field" that is
commonly superposed with coil-generated fields:

  B_ext(x) = B0 * (R0 / R) e_phi

where:
  - R0 is a reference major radius
  - B0 is the toroidal field magnitude at R=R0

For a purely toroidal field, field lines are circles at constant (R,Z) in cylindrical
coordinates. This makes it a clean sanity check for:

  - the field-line tracer
  - ParaView export of polylines
  - the 1/R magnitude scaling

Run:
  python examples/fieldline_tracing/external_toroidal_fieldlines.py
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

from torus_solver import ideal_toroidal_field, make_torus_surface
from torus_solver.fieldline import trace_field_lines_batch
from torus_solver.paraview import fieldlines_to_vtu, point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu
from torus_solver.plotting import ensure_dir, plot_fieldlines_3d, savefig, set_plot_style


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0, help="Reference major radius [m]")
    p.add_argument("--a", type=float, default=0.6, help="Minor radius for a context torus [m]")
    p.add_argument("--B0", type=float, default=1.0, help="External toroidal field at R=R0 [T]")
    p.add_argument("--rho", type=float, default=0.5, help="Seed minor-radius fraction in (0,1)")
    p.add_argument("--n-lines", type=int, default=8, help="Number of seed points / field lines")
    p.add_argument("--fieldline-steps", type=int, default=800)
    p.add_argument("--fieldline-ds", type=float, default=0.02)
    p.add_argument("--outdir", type=str, default="figures/external_toroidal_fieldlines")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true", help="Disable ParaView (.vtu/.vtm) outputs")
    p.add_argument("--plot-stride", type=int, default=3)
    args = p.parse_args()

    if not (0.0 < args.rho < 1.0):
        raise SystemExit("--rho must be in (0,1).")
    if args.n_lines <= 0:
        raise SystemExit("--n-lines must be positive.")

    outdir = None
    if (not args.no_plots) or (not args.no_paraview):
        outdir = ensure_dir(args.outdir)
        if not args.no_plots:
            set_plot_style()
            ensure_dir(outdir / "fieldlines")
            ensure_dir(outdir / "field")
            ensure_dir(outdir / "geometry")
        if not args.no_paraview:
            ensure_dir(outdir / "paraview")
        print(f"Output directory: {outdir}")

    print("External ideal toroidal field")
    print(f"  B_ext(R) = B0 * (R0/R) e_phi, with B0={args.B0} T at R0={args.R0} m")
    print(f"  tracing: n_lines={args.n_lines} steps={args.fieldline_steps} ds={args.fieldline_ds}")

    # Use a torus surface only as geometry context for plots / ParaView.
    surf = make_torus_surface(R0=float(args.R0), a=float(args.a), n_theta=64, n_phi=64)

    # Seed a poloidal ring at phi=0, on a minor-radius circle at rho*a.
    theta0 = jnp.linspace(0.0, 2 * jnp.pi, int(args.n_lines), endpoint=False)
    rho = float(args.rho) * float(args.a)
    R_seed = float(args.R0) + rho * jnp.cos(theta0)
    Z_seed = rho * jnp.sin(theta0)
    seeds = jnp.stack([R_seed, jnp.zeros_like(R_seed), Z_seed], axis=-1)

    def B_fn(xyz: jnp.ndarray) -> jnp.ndarray:
        return ideal_toroidal_field(xyz, B0=float(args.B0), R0=float(args.R0))

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
    traj_np = np.asarray(traj)  # (n_steps+1, n_lines, 3)

    # Sanity diagnostics: R and Z should be constant (up to integrator error) for each line.
    x = traj_np[..., 0]
    y = traj_np[..., 1]
    z = traj_np[..., 2]
    R = np.sqrt(x * x + y * y)
    dR = np.max(R, axis=0) - np.min(R, axis=0)
    dZ = np.max(z, axis=0) - np.min(z, axis=0)
    print(f"  invariance: max(ΔR)={float(np.max(dR)):.3e} m  max(ΔZ)={float(np.max(dZ)):.3e} m")

    # Sample B magnitude along the first line to check ~1/R scaling.
    sample = jnp.asarray(traj_np[:: max(int(args.fieldline_steps // 32), 1), 0, :], dtype=jnp.float64)
    B_samp = B_fn(sample)
    Bmag = jnp.linalg.norm(B_samp, axis=-1)
    R_samp = jnp.sqrt(sample[:, 0] ** 2 + sample[:, 1] ** 2)
    B_expected = float(args.B0) * float(args.R0) / R_samp
    rel = (Bmag - B_expected) / (B_expected + 1e-30)
    print(f"  1/R check: max_rel_err={float(jnp.max(jnp.abs(rel))):.3e}")

    if not args.no_paraview:
        assert outdir is not None
        pv_dir = ensure_dir(outdir / "paraview")

        surf_vtu = write_vtu(pv_dir / "torus_context.vtu", torus_surface_to_vtu(surface=surf))

        # ParaView polyline export expects (n_lines, n_points, 3).
        traj_pv = np.transpose(traj_np, (1, 0, 2))
        fl_vtu = write_vtu(
            pv_dir / "fieldlines.vtu",
            fieldlines_to_vtu(
                traj=traj_pv,
                point_data={
                    "line_id": np.repeat(np.arange(traj_pv.shape[0], dtype=int), traj_pv.shape[1]),
                },
            ),
        )

        samp_np = np.asarray(sample, dtype=float)
        sample_vtu = write_vtu(
            pv_dir / "sample_points.vtu",
            point_cloud_to_vtu(
                points=samp_np,
                point_data={
                    "|B|": np.asarray(Bmag, dtype=float),
                    "B_expected_1_over_R": np.asarray(B_expected, dtype=float),
                    "rel_err": np.asarray(rel, dtype=float),
                },
            ),
        )

        scene = write_vtm(
            pv_dir / "scene.vtm",
            {"torus_context": surf_vtu.name, "fieldlines": fl_vtu.name, "sample_points": sample_vtu.name},
        )
        print(f"ParaView scene: {scene}")

    if args.no_plots:
        return

    assert outdir is not None

    plot_fieldlines_3d(
        torus_xyz=np.asarray(surf.r),
        traj=traj_np,
        stride=int(args.plot_stride),
        line_stride=2,
        title="External ideal toroidal field lines (B ~ 1/R)",
        path=outdir / "fieldlines" / "fieldlines_3d.png",
    )

    # Simple 1/R plot.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(np.asarray(R_samp), np.asarray(Bmag), "o-", label=r"$|B|$ sampled")
    ax.plot(np.asarray(R_samp), np.asarray(B_expected), "k--", label=r"$B_0 R_0 / R$")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("|B| [T]")
    ax.set_title("External toroidal field magnitude scaling")
    ax.legend(loc="best")
    savefig(fig, outdir / "field" / "Bmag_vs_R.png", dpi=int(args.dpi))


if __name__ == "__main__":
    main()
