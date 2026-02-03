#!/usr/bin/env python3
"""Example: JAX vs NumPy for Biot–Savart evaluation on a winding surface.

This script exists to highlight two practical advantages of the JAX approach used in
this repository:

1) Performance: `jax.jit` can accelerate large, vectorized Biot–Savart evaluations.
2) Differentiability: the same code path is compatible with automatic differentiation
   (used throughout the optimization examples).

We compare:
  - a simple NumPy baseline that loops over evaluation points (typical "plain Python")
  - the JAX implementation `torus_solver.biot_savart_surface`, evaluated through a JIT-compiled function

Run:
  python examples/performance/jax_vs_numpy_biot_savart_speed.py
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

from torus_solver import biot_savart_surface, make_torus_surface
from torus_solver.biot_savart import MU0
from torus_solver.paraview import point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu
from torus_solver.plotting import ensure_dir, plot_3d_torus, plot_surface_map, savefig, set_plot_style
import torus_solver.plotting as tplot


def biot_savart_surface_numpy(
    *,
    r_surf: np.ndarray,  # (M,3)
    K_surf: np.ndarray,  # (M,3)
    dA: np.ndarray,  # (M,)
    eval_points: np.ndarray,  # (N,3)
    mu0: float,
    eps: float,
) -> np.ndarray:
    """Plain NumPy baseline (loops over evaluation points).

    This is intentionally kept "simple and obvious", as many legacy research codes are.
    It is reasonably vectorized over surface elements, but not over evaluation points.
    """
    pref = float(mu0) / (4.0 * np.pi)
    eps2 = float(eps) * float(eps)
    out = np.zeros((eval_points.shape[0], 3), dtype=float)

    for i in range(eval_points.shape[0]):
        P = eval_points[i]
        R = P[None, :] - r_surf  # (M,3)
        r2 = np.sum(R * R, axis=-1) + eps2  # (M,)
        inv_r3 = r2 ** (-1.5)
        dB = np.cross(K_surf, R) * inv_r3[:, None] * dA[:, None]
        out[i] = pref * np.sum(dB, axis=0)

    return out


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0)
    p.add_argument("--a", type=float, default=0.6)
    p.add_argument("--n-theta", type=int, default=64)
    p.add_argument("--n-phi", type=int, default=64)
    p.add_argument("--Ktheta", type=float, default=5e4, help="Constant poloidal current density [A/m]")
    p.add_argument("--n-eval", type=int, default=256, help="Number of evaluation points")
    p.add_argument("--rho-eval", type=float, default=0.55, help="Minor-radius fraction for evaluation points")
    p.add_argument("--eps", type=float, default=1e-8, help="Biot–Savart softening length [m]")
    p.add_argument("--chunk-size", type=int, default=256, help="JAX chunk size (None disables chunking)")
    p.add_argument("--repeat", type=int, default=10, help="Timing repeats (post-compilation)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="figures/jax_vs_numpy_biot_savart_speed")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true")
    p.add_argument("--plot-stride", type=int, default=3)
    args = p.parse_args()

    if not (0.0 < args.rho_eval < 0.99):
        raise SystemExit("--rho-eval must be in (0,1).")
    if args.repeat <= 0:
        raise SystemExit("--repeat must be positive.")

    outdir = None
    if (not args.no_plots) or (not args.no_paraview):
        outdir = ensure_dir(args.outdir)
        if not args.no_plots:
            set_plot_style()
            ensure_dir(outdir / "geometry")
            ensure_dir(outdir / "maps")
            ensure_dir(outdir / "timing")
        if not args.no_paraview:
            ensure_dir(outdir / "paraview")

    surface = make_torus_surface(R0=float(args.R0), a=float(args.a), n_theta=int(args.n_theta), n_phi=int(args.n_phi))

    # Simple current sheet: constant K along +θ.
    e_theta = surface.r_theta / surface.a
    K = float(args.Ktheta) * e_theta

    # Random evaluation points on a poloidal circle at φ=0 (inside the torus).
    rng = np.random.default_rng(int(args.seed))
    theta = rng.uniform(0.0, 2 * np.pi, size=(int(args.n_eval),))
    rho = float(args.rho_eval) * float(args.a)
    R = args.R0 + rho * np.cos(theta)
    Z = rho * np.sin(theta)
    eval_np = np.stack([R, np.zeros_like(R), Z], axis=-1).astype(float)
    eval_jax = jnp.asarray(eval_np, dtype=jnp.float64)

    # ---- NumPy baseline ----
    r_surf = np.asarray(surface.r, dtype=float).reshape((-1, 3))
    K_surf = np.asarray(K, dtype=float).reshape((-1, 3))
    dA = np.asarray(surface.area_weights, dtype=float).reshape((-1,))

    t0 = time.perf_counter()
    B_np = biot_savart_surface_numpy(
        r_surf=r_surf, K_surf=K_surf, dA=dA, eval_points=eval_np, mu0=float(MU0), eps=float(args.eps)
    )
    t1 = time.perf_counter()

    # ---- JAX (jit) ----
    chunk_size = None if int(args.chunk_size) <= 0 else int(args.chunk_size)

    def B_jit_fn(pts: jnp.ndarray) -> jnp.ndarray:
        return biot_savart_surface(surface, K, pts, eps=float(args.eps), chunk_size=chunk_size)

    B_jit = jax.jit(B_jit_fn)

    # Compile / warmup.
    _ = B_jit(eval_jax).block_until_ready()

    t2 = time.perf_counter()
    for _k in range(int(args.repeat)):
        B_j = B_jit(eval_jax)
    B_j.block_until_ready()
    t3 = time.perf_counter()
    B_jax = np.asarray(B_j, dtype=float)

    # Accuracy check (should match closely in double precision).
    max_abs = float(np.max(np.abs(B_jax - B_np)))
    rel = float(max_abs / (np.max(np.abs(B_np)) + 1e-30))

    t_numpy = t1 - t0
    t_jax = (t3 - t2) / float(args.repeat)
    speedup = t_numpy / t_jax if t_jax > 0 else np.inf

    print("Biot–Savart speed comparison")
    print(f"  surface grid: n_theta={args.n_theta} n_phi={args.n_phi}  (M={r_surf.shape[0]})")
    print(f"  eval points:  n_eval={args.n_eval}")
    print(f"  eps={args.eps:g}  chunk_size={chunk_size}")
    print(f"  NumPy baseline (1x): {t_numpy:.3f} s")
    print(f"  JAX jit (avg over {args.repeat}x): {t_jax:.4f} s/call   speedup={speedup:.1f}x")
    print(f"  agreement: max_abs={max_abs:.3e} T   rel={rel:.3e}")

    if outdir is None:
        return

    # ParaView outputs.
    if not args.no_paraview:
        pv = ensure_dir(outdir / "paraview")
        Kmag = np.asarray(jnp.linalg.norm(K, axis=-1)).reshape(-1)
        surf_vtu = write_vtu(
            pv / "winding_surface.vtu",
            torus_surface_to_vtu(
                surface=surface,
                point_data={
                    "K": np.asarray(K).reshape(-1, 3),
                    "|K|": Kmag,
                    "Ktheta_const_A_per_m": np.full_like(Kmag, float(args.Ktheta)),
                },
            ),
        )
        eval_vtu = write_vtu(
            pv / "eval_points.vtu",
            point_cloud_to_vtu(
                points=eval_np,
                point_data={
                    "B_numpy": B_np,
                    "B_jax": B_jax,
                    "|B|_numpy": np.linalg.norm(B_np, axis=-1),
                    "|B|_jax": np.linalg.norm(B_jax, axis=-1),
                },
            ),
        )
        scene = write_vtm(pv / "scene.vtm", {"winding_surface": surf_vtu.name, "eval_points": eval_vtu.name})
        print(f"ParaView scene: {scene}")

    if args.no_plots:
        return

    # Plots.
    # 1) 3D geometry context.
    plot_3d_torus(
        torus_xyz=np.asarray(surface.r),
        electrodes_xyz=None,
        curve_xyz=None,
        eval_xyz=eval_np,
        stride=int(args.plot_stride),
        title="Torus + evaluation points (for Biot–Savart timing)",
        path=outdir / "geometry" / "torus_and_eval.png",
    )

    # 2) Surface map of |K|.
    plot_surface_map(
        phi=np.asarray(surface.phi),
        theta=np.asarray(surface.theta),
        data=np.asarray(jnp.linalg.norm(K, axis=-1)),
        title=r"Winding surface: $|K|$ (constant $K_\\theta$ case)",
        cbar_label=r"$|K|$ [A/m]",
        path=outdir / "maps" / "Kmag.png",
    )

    # 3) Timing summary bar plot.
    fig, ax = tplot.plt.subplots(constrained_layout=True, figsize=(6.2, 4.6))
    ax.bar(["NumPy\n(1x)"], [t_numpy], color="0.6", label="NumPy baseline")
    ax.bar([f"JAX jit\n(avg {args.repeat}x)"], [t_jax], color="tab:blue", label="JAX jit")
    ax.set_ylabel("Wall time [s]")
    ax.set_title(f"Biot–Savart evaluation time (speedup={speedup:.1f}x)")
    ax.legend(loc="best")
    savefig(fig, outdir / "timing" / "timing.png", dpi=int(args.dpi))


if __name__ == "__main__":
    main()
