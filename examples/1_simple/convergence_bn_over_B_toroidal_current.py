#!/usr/bin/env python3
"""Example: convergence of max|Bn/B| for an axisymmetric toroidal field on an interior torus.

We prescribe a *net poloidal current* on the winding surface using the current-potential
model (REGCOIL-style):

  K = n_hat × ∇_s Phi,

with an additional multi-valued secular term that produces a net poloidal current
I_pol (Amperes):

  Phi_sec = (I_pol / 2π) * φ  ->  ∂_φ Phi adds I_pol/(2π).

This generates an approximately *toroidal* magnetic field inside the torus, and for a
purely toroidal field we should have:

  B·n = 0

on any *interior* torus surface (minor radius < winding-surface minor radius).

We sweep winding-surface resolution and monitor:

  - max |Bn/|B|| on the interior target surface
  - RMS(Bn/|B|) on the interior target surface
  - max relative error of B_phi(R) against Ampère's law:
        B_phi(R) ≈ μ0 I_pol / (2π R)

Run:
  python examples/1_simple/convergence_bn_over_B_toroidal_current.py
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
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from torus_solver import biot_savart_surface, make_torus_surface, surface_current_from_current_potential_with_net_currents
from torus_solver.biot_savart import MU0
from torus_solver.fields import cylindrical_coords, cylindrical_unit_vectors
from torus_solver.metrics import bn_over_B_metrics
from torus_solver.paraview import point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu
from torus_solver.plotting import ensure_dir, plot_surface_map, savefig, set_plot_style
from torus_solver.targets import circular_torus_target_surface

import torus_solver.plotting as tplot


@dataclass(frozen=True)
class SweepResult:
    n_theta: int
    n_phi: int
    n_grid: int
    bn_over_B_rms: float
    bn_over_B_max: float
    Bphi_rel_err_max: float


def _parse_resolutions(spec: str) -> list[tuple[int, int]]:
    items = []
    for chunk in (spec or "").split(","):
        s = chunk.strip()
        if not s:
            continue
        if "x" not in s:
            raise ValueError(f"Bad --resolutions entry '{s}'. Use like '32x48,64x96'.")
        a, b = s.split("x", 1)
        items.append((int(a), int(b)))
    if not items:
        raise ValueError("No resolutions provided.")
    return items


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0, help="Winding-surface major radius [m]")
    p.add_argument("--a", type=float, default=0.3, help="Winding-surface minor radius [m]")
    p.add_argument("--Ipol", type=float, default=2.0e6, help="Net poloidal current on winding surface [A]")
    p.add_argument("--rho-target-frac", type=float, default=0.5, help="Interior target minor radius as fraction of a")
    p.add_argument("--target-n-theta", type=int, default=18)
    p.add_argument("--target-n-phi", type=int, default=22)
    p.add_argument(
        "--resolutions",
        type=str,
        default="24x32,32x48,48x64,64x96",
        help="Comma-separated list like '24x32,64x96' (winding surface θ×φ).",
    )
    p.add_argument("--midplane-nR", type=int, default=41, help="Number of midplane R samples for Bphi check")
    p.add_argument("--chunk-size", type=int, default=256, help="Chunk size for Biot–Savart evaluation points")
    p.add_argument("--outdir", type=str, default="figures/convergence_bn_over_B_toroidal_current")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true", help="Disable ParaView (.vtu/.vtm) outputs")
    args = p.parse_args()

    if not (0.0 < args.rho_target_frac < 1.0):
        raise SystemExit("--rho-target-frac must be in (0,1).")

    resolutions = _parse_resolutions(args.resolutions)
    rho_target = float(args.rho_target_frac) * float(args.a)

    outdir = None
    if (not args.no_plots) or (not args.no_paraview):
        outdir = ensure_dir(args.outdir)
        if not args.no_plots:
            set_plot_style()
            ensure_dir(outdir / "field")
            ensure_dir(outdir / "convergence")
            ensure_dir(outdir / "maps")
        if not args.no_paraview:
            ensure_dir(outdir / "paraview")
        print(f"Output directory: {outdir}")

    print("Convergence sweep: axisymmetric toroidal field from net poloidal current")
    print(f"  winding surface: R0={args.R0} m, a={args.a} m")
    print(f"  target surface:  a_target=rho_target_frac*a = {args.rho_target_frac}*{args.a} = {rho_target} m")
    print(f"  Ipol={args.Ipol:.6e} A")
    print(f"  target grid: n_theta={args.target_n_theta}, n_phi={args.target_n_phi}")
    print(f"  winding resolutions: {', '.join([f'{nt}x{np_}' for nt, np_ in resolutions])}")

    target = circular_torus_target_surface(
        R0=float(args.R0), a=float(rho_target), n_theta=int(args.target_n_theta), n_phi=int(args.target_n_phi)
    )
    target_pts = target.xyz.reshape((-1, 3))
    target_n = target.normals.reshape((-1, 3))
    target_w = target.weights.reshape((-1,))

    # Midplane points for Ampère-law comparison at phi=0:
    R_eval = jnp.linspace(float(args.R0) - 0.95 * rho_target, float(args.R0) + 0.95 * rho_target, int(args.midplane_nR))
    pts_mid = jnp.stack([R_eval, jnp.zeros_like(R_eval), jnp.zeros_like(R_eval)], axis=-1)
    _, phi_mid, _ = cylindrical_coords(pts_mid)
    _, e_phi_mid, _ = cylindrical_unit_vectors(phi_mid)
    Bphi_expected = float(MU0) * float(args.Ipol) / (2 * np.pi * R_eval)

    results: list[SweepResult] = []

    for n_theta, n_phi in resolutions:
        print(f"\n--- resolution: n_theta={n_theta} n_phi={n_phi} ---")
        surface = make_torus_surface(R0=float(args.R0), a=float(args.a), n_theta=int(n_theta), n_phi=int(n_phi))
        Phi_sv = jnp.zeros((surface.theta.size, surface.phi.size), dtype=jnp.float64)
        K = surface_current_from_current_potential_with_net_currents(surface, Phi_sv, net_poloidal_current_A=float(args.Ipol))

        B_target = biot_savart_surface(surface, K, target_pts, eps=1e-9, chunk_size=int(args.chunk_size))
        ratio, rms, max_abs = bn_over_B_metrics(B_target, target_n, target_w)
        ratio.block_until_ready()

        B_mid = biot_savart_surface(surface, K, pts_mid, eps=1e-9, chunk_size=int(args.chunk_size))
        Bphi_mid = jnp.sum(B_mid * e_phi_mid, axis=-1)
        rel_err_mid = jnp.abs(Bphi_mid - Bphi_expected) / (jnp.abs(Bphi_expected) + 1e-30)
        rel_err_max = float(jnp.max(rel_err_mid))

        res = SweepResult(
            n_theta=int(n_theta),
            n_phi=int(n_phi),
            n_grid=int(n_theta) * int(n_phi),
            bn_over_B_rms=float(rms),
            bn_over_B_max=float(max_abs),
            Bphi_rel_err_max=float(rel_err_max),
        )
        results.append(res)

        print(f"  max|Bn/B| = {res.bn_over_B_max:.3e}   rms(Bn/B) = {res.bn_over_B_rms:.3e}")
        print(f"  max rel error in Bphi vs μ0 Ipol/(2πR): {res.Bphi_rel_err_max:.3e}")

        # Save a representative ParaView scene for the finest grid only.
        if (not args.no_paraview) and outdir is not None and (n_theta, n_phi) == resolutions[-1]:
            pv_dir = ensure_dir(outdir / "paraview")

            K_np = np.asarray(K).reshape(-1, 3)
            Kmag = np.linalg.norm(K_np, axis=-1)
            surf_pd = {"K": K_np, "|K|": Kmag, "Ipol_A": np.full_like(Kmag, float(args.Ipol))}
            surf_vtu = write_vtu(pv_dir / "winding_surface.vtu", torus_surface_to_vtu(surface=surface, point_data=surf_pd))

            B_np = np.asarray(B_target)
            ratio_np = np.asarray(ratio)
            Bmag_np = np.linalg.norm(B_np, axis=-1)
            tgt_pd = {
                "B": B_np,
                "|B|": Bmag_np,
                "Bn_over_B": ratio_np,
                "n_hat": np.asarray(target_n),
            }
            tgt_vtu = write_vtu(
                pv_dir / "target_points.vtu",
                point_cloud_to_vtu(points=np.asarray(target_pts), point_data=tgt_pd),
            )

            mid_pd = {
                "B": np.asarray(B_mid),
                "Bphi_expected": np.asarray(Bphi_expected),
                "Bphi": np.asarray(Bphi_mid),
                "rel_err_Bphi": np.asarray(rel_err_mid),
            }
            mid_vtu = write_vtu(
                pv_dir / "midplane_points.vtu",
                point_cloud_to_vtu(points=np.asarray(pts_mid), point_data=mid_pd),
            )

            scene = write_vtm(
                pv_dir / "scene.vtm",
                {"winding_surface": surf_vtu.name, "target_points": tgt_vtu.name, "midplane_points": mid_vtu.name},
            )
            print(f"  ParaView scene: {scene}")

        # Save a 2D Bn/B map for the finest grid.
        if (not args.no_plots) and outdir is not None and (n_theta, n_phi) == resolutions[-1]:
            ratio_grid = np.asarray(ratio).reshape((int(args.target_n_theta), int(args.target_n_phi)))
            plot_surface_map(
                phi=np.asarray(target.phi),
                theta=np.asarray(target.theta),
                data=ratio_grid,
                title=r"$B_n/|B|$ on interior target torus (should be ~0)",
                cbar_label=r"$B_n/|B|$",
                path=outdir / "maps" / "Bn_over_B_map.png",
                cmap="coolwarm",
                vmin=-0.02,
                vmax=0.02,
            )

            # Bphi midplane comparison plot.
            fig, ax = tplot.plt.subplots(constrained_layout=True)
            ax.plot(np.asarray(R_eval), np.asarray(Bphi_expected), "k--", label=r"$\mu_0 I_{\mathrm{pol}}/(2\pi R)$")
            ax.plot(np.asarray(R_eval), np.asarray(Bphi_mid), color="tab:blue", label="Biot–Savart")
            ax.set_xlabel(r"$R$ [m]")
            ax.set_ylabel(r"$B_\phi$ [T]")
            ax.set_title(r"Midplane $B_\phi(R)$: Biot–Savart vs Ampère-law $1/R$")
            ax.legend()
            savefig(fig, outdir / "field" / "Bphi_vs_R.png", dpi=int(args.dpi))

    # Summary convergence plot.
    if (not args.no_plots) and outdir is not None:
        n_grid = np.asarray([r.n_grid for r in results], dtype=int)
        bn_max = np.asarray([r.bn_over_B_max for r in results], dtype=float)
        bn_rms = np.asarray([r.bn_over_B_rms for r in results], dtype=float)
        bphi_rel = np.asarray([r.Bphi_rel_err_max for r in results], dtype=float)

        fig, axs = tplot.plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

        ax = axs[0]
        ax.plot(n_grid, bn_max, "o-", label=r"max$|B_n/|B||$")
        ax.plot(n_grid, bn_rms, "s-", label=r"RMS$(B_n/|B|)$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_\theta N_\phi$ (winding surface)")
        ax.set_ylabel(r"$B_n/|B|$")
        ax.set_title(r"Interior normal-field convergence")
        ax.legend()

        ax = axs[1]
        ax.plot(n_grid, bphi_rel, "o-", color="tab:purple")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$N_\theta N_\phi$ (winding surface)")
        ax.set_ylabel("max rel error")
        ax.set_title(r"Midplane $B_\phi$ vs $\mu_0 I_{\mathrm{pol}}/(2\pi R)$")

        savefig(fig, outdir / "convergence" / "convergence_summary.png", dpi=int(args.dpi))


if __name__ == "__main__":
    main()

