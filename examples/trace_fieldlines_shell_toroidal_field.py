#!/usr/bin/env python3
"""Example: shell poloidal current sheet -> toroidal 1/R field + field-line tracing.

We build a torus surface and prescribe a uniform surface current density K along the
poloidal direction (θ). This is the continuous analogue of a tightly-wound toroidal
solenoid, producing an approximately toroidal field:

  B ≈ μ0 Kθ (R0/R) e_φ

We trace field lines using the analytic ideal field (fast), then validate by
sampling the Biot–Savart field from the surface current sheet along the traced line.

Run:
  - `python examples/trace_fieldlines_shell_toroidal_field.py`
"""

from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from torus_solver import biot_savart_surface, ideal_toroidal_field, make_torus_surface
from torus_solver.biot_savart import MU0
from torus_solver.fieldline import trace_field_line
from torus_solver.fields import cylindrical_coords, cylindrical_unit_vectors
from torus_solver.plotting import ensure_dir, plot_3d_torus, plot_fieldline, savefig, set_plot_style
import torus_solver.plotting as tplot


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0)
    p.add_argument("--a", type=float, default=0.3)
    p.add_argument("--n-theta", type=int, default=96)
    p.add_argument("--n-phi", type=int, default=128)
    p.add_argument("--B0", type=float, default=1e-3, help="Toroidal field at R=R0 (T)")
    p.add_argument("--n", type=int, default=800, help="Trace steps")
    p.add_argument("--outdir", type=str, default="figures/trace_fieldlines_shell_toroidal_field")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plot-stride", type=int, default=3)
    args = p.parse_args()

    if not args.no_plots:
        set_plot_style()
        outdir = ensure_dir(args.outdir)
        ensure_dir(outdir / "traj")
        ensure_dir(outdir / "field")
        ensure_dir(outdir / "geometry")
        print(f"Saving figures to: {outdir}")

    surface = make_torus_surface(R0=args.R0, a=args.a, n_theta=args.n_theta, n_phi=args.n_phi)

    # Choose Kθ so that Bφ(R0) ≈ B0. Sign convention depends on (θ,φ) orientation.
    Ktheta = -args.B0 / float(MU0)  # A/m
    e_theta = surface.r_theta / surface.a
    K = Ktheta * e_theta

    print("Shell toroidal-field setup")
    print(f"  R0={args.R0} a={args.a} n_theta={args.n_theta} n_phi={args.n_phi}")
    print(f"  target B0={args.B0} T  -> Ktheta={Ktheta:+.6e} A/m")

    # Trace using the analytic ideal toroidal field.
    r0 = jnp.array([args.R0 + 0.5 * args.a, 0.0, 0.0], dtype=jnp.float64)

    def B_analytic(xyz):
        return ideal_toroidal_field(xyz, B0=args.B0, R0=args.R0)

    pts = jax.jit(lambda x: trace_field_line(B_analytic, x, step_size=0.01, n_steps=args.n))(r0)

    # Validate Biot–Savart along a decimated set of points (to keep runtime small).
    idx = jnp.linspace(0, pts.shape[0] - 1, 64, dtype=jnp.int32)
    sample = pts[idx]
    B_bs = biot_savart_surface(surface, K, sample, eps=1e-9)
    B_an = B_analytic(sample)

    err = jnp.linalg.norm(B_bs - B_an, axis=-1)
    rel = err / (jnp.linalg.norm(B_an, axis=-1) + 1e-30)

    print("Trace summary")
    print(f"  start r0={list(map(float, r0))}")
    print(f"  traced_points={pts.shape[0]} sampled={sample.shape[0]}")
    print(f"  |B_an| mean={float(jnp.mean(jnp.linalg.norm(B_an, axis=-1))):.3e} T")
    print(f"  Biot–Savart vs analytic: max_abs={float(jnp.max(err)):.3e} T  max_rel={float(jnp.max(rel)):.3e}")

    if not args.no_plots:
        pts_np = np.asarray(pts)
        sample_np = np.asarray(sample)

        plot_fieldline(
            pts=pts_np,
            title="Ideal toroidal field line (analytic)",
            R0=args.R0,
            path3d=outdir / "traj" / "fieldline_3d.png",
            path_rz=outdir / "traj" / "fieldline_RZ.png",
        )

        # 3D geometry: torus + traced line + sampled points.
        plot_3d_torus(
            torus_xyz=np.asarray(surface.r),
            curve_xyz=pts_np,
            eval_xyz=sample_np,
            stride=args.plot_stride,
            title="Torus + analytic field line + sampled points",
            path=outdir / "geometry" / "torus_and_fieldline.png",
        )

        # Compare toroidal component along the sample points.
        R, phi, _ = cylindrical_coords(sample)
        _, e_phi, _ = cylindrical_unit_vectors(phi)
        Bphi_an = np.sum(np.asarray(B_an) * np.asarray(e_phi), axis=-1)
        Bphi_bs = np.sum(np.asarray(B_bs) * np.asarray(e_phi), axis=-1)
        s = np.arange(sample_np.shape[0])  # sample index as proxy arc parameter

        fig, ax = tplot.plt.subplots(constrained_layout=True)
        ax.plot(s, Bphi_an, "k--", label="analytic")
        ax.plot(s, Bphi_bs, color="tab:blue", label="Biot–Savart")
        ax.set_xlabel("Sample index")
        ax.set_ylabel(r"$B_\phi$ [T]")
        ax.set_title(r"Toroidal field along traced line: $B_\phi$ comparison")
        ax.legend()
        savefig(fig, outdir / "field" / "Bphi_comparison.png", dpi=args.dpi)

        fig, ax = tplot.plt.subplots(constrained_layout=True)
        abs_err = np.abs(Bphi_bs - Bphi_an)
        rel_err = abs_err / (np.abs(Bphi_an) + 1e-30)
        ax.plot(s, abs_err, label="abs")
        ax.plot(s, rel_err, label="rel")
        ax.set_xlabel("Sample index")
        ax.set_yscale("log")
        ax.set_title("Toroidal-component error (log scale)")
        ax.legend()
        savefig(fig, outdir / "field" / "Bphi_error.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
