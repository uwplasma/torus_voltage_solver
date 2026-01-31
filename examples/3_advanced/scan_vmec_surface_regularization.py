#!/usr/bin/env python3
"""REGCOIL-style regularization scan for a VMEC target surface.

This script runs a sequence of *current-potential* optimizations on a circular torus
winding surface while sweeping the regularization weight on ⟨|K|^2⟩, producing an
"L-curve" / tradeoff curve between:

  - field-quality: max|Bn/B| and rms(Bn/B) on the target surface
  - current magnitude: K_rms on the winding surface

The continuation strategy (warm-starting each solve from the previous solution)
is commonly used in coil/current-potential design workflows.

Run:
  python examples/3_advanced/scan_vmec_surface_regularization.py --vmec-input examples/data/vmec/input.QA_nfp2
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
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jaxopt import LBFGS

from torus_solver import make_torus_surface
from torus_solver.biot_savart import MU0, biot_savart_surface
from torus_solver.current_potential import surface_current_from_current_potential_with_net_currents
from torus_solver.metrics import bn_over_B_metrics, weighted_p_norm
from torus_solver.paraview import point_cloud_to_vtu, torus_surface_to_vtu, write_vtm, write_vtu
from torus_solver.plotting import ensure_dir, plot_surface_map, savefig, set_plot_style
from torus_solver.targets import vmec_target_surface
from torus_solver.vmec import read_vmec_boundary


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PotentialParams:
    cos_mn: jnp.ndarray  # (M, N)
    sin_mn: jnp.ndarray  # (M, N)

    def tree_flatten(self):
        return (self.cos_mn, self.sin_mn), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        cos_mn, sin_mn = children
        return cls(cos_mn=cos_mn, sin_mn=sin_mn)


def _resolve_vmec_input(vmec_input: str) -> Path:
    p = Path(vmec_input)
    if p.exists():
        return p

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir
    while repo_root != repo_root.parent and not (repo_root / "pyproject.toml").exists():
        repo_root = repo_root.parent

    candidates = [
        script_dir / vmec_input,
        repo_root / vmec_input,
        repo_root / "examples" / vmec_input,
        repo_root / "examples" / "data" / "vmec" / Path(vmec_input).name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "VMEC input not found.\n"
        f"  got: {vmec_input}\n"
        f"  tried: {p.resolve()}\n"
        + "".join(f"  tried: {c}\n" for c in candidates)
    )


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--vmec-input", type=str, default="examples/data/vmec/input.QA_nfp2")

    p.add_argument("--R0", type=float, default=1.0)
    p.add_argument("--a", type=float, default=0.3)
    p.add_argument("--n-theta", type=int, default=48)
    p.add_argument("--n-phi", type=int, default=48)

    p.add_argument("--surf-n-theta", type=int, default=32)
    p.add_argument("--surf-n-phi", type=int, default=56)
    p.add_argument("--fit-margin", type=float, default=0.7)

    p.add_argument("--B0", type=float, default=1.0, help="Toroidal field scale at R=R0 [T]")
    p.add_argument(
        "--net-poloidal-current",
        type=float,
        default=None,
        help="Net poloidal current [A]. If omitted, uses 2π R0 B0 / μ0.",
    )
    p.add_argument("--net-toroidal-current", type=float, default=0.0)
    p.add_argument("--phi-scale", type=float, default=None)

    p.add_argument("--mpol-potential", type=int, default=23)
    p.add_argument("--ntor-potential", type=int, default=11)
    p.add_argument("--bn-p", type=float, default=8.0)

    p.add_argument("--regK-min", type=float, default=1e-9)
    p.add_argument("--regK-max", type=float, default=1e-6)
    p.add_argument("--n-regK", type=int, default=7)

    p.add_argument("--lbfgs-maxiter", type=int, default=200)
    p.add_argument("--lbfgs-tol", type=float, default=1e-9)
    p.add_argument("--biot-savart-eps", type=float, default=1e-8)
    p.add_argument("--no-warm-start", action="store_true")

    p.add_argument("--outdir", type=str, default="figures/scan_vmec_surface_regularization")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true", help="Disable ParaView (.vtu/.vtm) outputs")
    args = p.parse_args()

    outdir = None
    if (not args.no_plots) or (not args.no_paraview):
        outdir = ensure_dir(args.outdir)
        if not args.no_plots:
            set_plot_style()
            ensure_dir(outdir / "maps")
            print(f"Saving figures to: {outdir}")
        if not args.no_paraview:
            ensure_dir(outdir / "paraview")
            print(f"Saving ParaView outputs to: {outdir / 'paraview'}")

    vmec_path = _resolve_vmec_input(args.vmec_input)
    print("Loading VMEC boundary:", str(vmec_path))
    boundary = read_vmec_boundary(str(vmec_path))
    print(f"  parsed NFP={boundary.nfp}  nmodes={boundary.m.size}")

    target = vmec_target_surface(
        boundary,
        torus_R0=float(args.R0),
        torus_a=float(args.a),
        fit_margin=float(args.fit_margin),
        n_theta=int(args.surf_n_theta),
        n_phi=int(args.surf_n_phi),
        dtype=jnp.float64,
    )
    print("Target surface fit into circular torus:")
    print(f"  torus: R0={args.R0} a={args.a} fit_margin={args.fit_margin}")
    print(
        "  shift_R={:+.6e} m  scale={:.6e}  rho_max(before)={:.6e} m".format(
            float(target.fit.shift_R), float(target.fit.scale_rho), float(target.fit.rho_max_before_m)
        )
    )

    eval_points = target.xyz.reshape((-1, 3))
    normals = target.normals.reshape((-1, 3))
    weights = target.weights.reshape((-1,))

    surface = make_torus_surface(R0=float(args.R0), a=float(args.a), n_theta=int(args.n_theta), n_phi=int(args.n_phi))

    nfp = int(boundary.nfp)
    mpol_req = int(args.mpol_potential)
    ntor_req = int(args.ntor_potential)
    mpol_max = max(int(surface.theta.size // 2 - 1), 0)
    ntor_max = max(int((surface.phi.size // 2 - 1) // max(nfp, 1)), 0)
    mpol = min(mpol_req, mpol_max)
    ntor = min(ntor_req, ntor_max)
    if mpol != mpol_req or ntor != ntor_req:
        print(
            "Clipping potential modes to avoid aliasing on the winding-surface grid:"
            f" requested (mpol={mpol_req}, ntor={ntor_req})"
            f" -> using (mpol={mpol}, ntor={ntor})"
            f" with n_theta={surface.theta.size}, n_phi={surface.phi.size}, nfp={nfp}"
        )

    if args.net_poloidal_current is None:
        if float(args.B0) == 0.0:
            net_poloidal_current_A = 0.0
        else:
            net_poloidal_current_A = float(2 * np.pi * args.R0 * args.B0 / float(MU0))
    else:
        net_poloidal_current_A = float(args.net_poloidal_current)
    net_toroidal_current_A = float(args.net_toroidal_current)
    if args.phi_scale is None:
        phi_scale_A = float(max(abs(net_poloidal_current_A), abs(net_toroidal_current_A), 1.0))
    else:
        phi_scale_A = float(args.phi_scale)

    print("Current-potential parameterization:")
    print(f"  mpol={mpol} ntor={ntor} nfp={nfp}")
    print(f"  Ipol={net_poloidal_current_A:+.3e} A  Itor={net_toroidal_current_A:+.3e} A")
    print(f"  Phi_scale={phi_scale_A:.3e} A")

    M = mpol + 1
    N = 2 * ntor + 1
    mask = jnp.ones((M, N), dtype=jnp.float64).at[0, ntor].set(0.0)

    th2 = surface.theta[None, None, :, None]
    ph2 = surface.phi[None, None, None, :]
    m_idx = jnp.arange(M, dtype=jnp.float64)[:, None, None, None]
    n_idx = (jnp.arange(-ntor, ntor + 1, dtype=jnp.float64) * float(nfp))[None, :, None, None]
    angle = m_idx * th2 - n_idx * ph2
    cosA = jnp.cos(angle)
    sinA = jnp.sin(angle)

    def Phi_from_params(pot: PotentialParams) -> jnp.ndarray:
        Phi_unit = jnp.sum(
            (pot.cos_mn * mask)[:, :, None, None] * cosA + (pot.sin_mn * mask)[:, :, None, None] * sinA,
            axis=(0, 1),
        )
        return phi_scale_A * Phi_unit

    K_scale = 1.0
    if net_poloidal_current_A != 0.0:
        K_scale = max(K_scale, abs(net_poloidal_current_A) / (2 * np.pi * max(float(args.R0), 1e-12)))
    if net_toroidal_current_A != 0.0:
        K_scale = max(K_scale, abs(net_toroidal_current_A) / (2 * np.pi * max(float(args.a), 1e-12)))

    def loss_fn(pot: PotentialParams, *, reg_K: float):
        Phi_sv = Phi_from_params(pot)
        K = surface_current_from_current_potential_with_net_currents(
            surface,
            Phi_sv,
            net_poloidal_current_A=net_poloidal_current_A,
            net_toroidal_current_A=net_toroidal_current_A,
        )
        B = biot_savart_surface(surface, K, eval_points, eps=float(args.biot_savart_eps))
        ratio, rms, max_abs = bn_over_B_metrics(B, normals, weights)
        bn_metric = weighted_p_norm(ratio, weights, p=float(args.bn_p))
        loss_bn = bn_metric * bn_metric

        K2 = jnp.sum(K * K, axis=-1)
        mean_K2 = jnp.sum(K2 * surface.area_weights) / (jnp.sum(surface.area_weights) + 1e-30)
        loss_reg = float(reg_K) * (mean_K2 / (K_scale * K_scale))
        loss = loss_bn + loss_reg
        aux = {
            "loss": loss,
            "loss_bn": loss_bn,
            "loss_reg": loss_reg,
            "Bn_over_B_rms": rms,
            "Bn_over_B_max": max_abs,
            "Bn_over_B_p": bn_metric,
            "K_rms": jnp.sqrt(mean_K2),
        }
        return loss, aux

    # Regularization scan (descending reg_K for continuation).
    regKs = np.geomspace(float(args.regK_max), float(args.regK_min), int(args.n_regK))

    params0 = PotentialParams(
        cos_mn=jnp.zeros((M, N), dtype=jnp.float64),
        sin_mn=jnp.zeros((M, N), dtype=jnp.float64),
    )

    rows: list[dict[str, float]] = []
    last = params0
    for i, regK in enumerate(regKs):
        if args.no_warm_start:
            init = params0
        else:
            init = last

        solver = LBFGS(
            fun=lambda pot: loss_fn(pot, reg_K=float(regK)),
            has_aux=True,
            maxiter=int(args.lbfgs_maxiter),
            tol=float(args.lbfgs_tol),
            jit=True,
        )

        print(f"[{i+1}/{regKs.size}] reg_K={regK:.3e} ...")
        t0 = time.perf_counter()
        res = solver.run(init)
        t1 = time.perf_counter()

        last = res.params
        aux = res.state.aux
        row = {
            "reg_K": float(regK),
            "loss": float(aux["loss"]),
            "loss_bn": float(aux["loss_bn"]),
            "loss_reg": float(aux["loss_reg"]),
            "Bn_over_B_rms": float(aux["Bn_over_B_rms"]),
            "Bn_over_B_max": float(aux["Bn_over_B_max"]),
            "Bn_over_B_p": float(aux["Bn_over_B_p"]),
            "K_rms_A_per_m": float(aux["K_rms"]),
            "wall_time_s": float(t1 - t0),
            "iters": float(res.state.iter_num),
        }
        rows.append(row)
        print(
            "  iters={:4.0f} wall_time={:.2f}s  rms(Bn/B)={:.3e}  max|Bn/B|={:.3e}  K_rms={:.3e} A/m".format(
                row["iters"],
                row["wall_time_s"],
                row["Bn_over_B_rms"],
                row["Bn_over_B_max"],
                row["K_rms_A_per_m"],
            )
        )

    # Save a CSV + NPZ artifact for post-processing.
    out_npz = None
    if outdir is not None:
        out_npz = Path(outdir) / "regularization_scan.npz"
        cols = {k: np.array([r[k] for r in rows], dtype=float) for k in rows[0].keys()}
        np.savez(out_npz, **cols)
        print("Saved scan data:", out_npz)

        out_csv = Path(outdir) / "regularization_scan.csv"
        header = ",".join(cols.keys())
        data = np.stack([cols[k] for k in cols.keys()], axis=1)
        np.savetxt(out_csv, data, delimiter=",", header=header, comments="")
        print("Saved scan table:", out_csv)

    # Best (smallest reg_K) solution in the scan.
    best_params = last
    Phi_sv = Phi_from_params(best_params)
    K = surface_current_from_current_potential_with_net_currents(
        surface,
        Phi_sv,
        net_poloidal_current_A=net_poloidal_current_A,
        net_toroidal_current_A=net_toroidal_current_A,
    )
    B = biot_savart_surface(surface, K, eval_points, eps=float(args.biot_savart_eps))
    ratio, rms, max_abs = bn_over_B_metrics(B, normals, weights)
    ratio_grid = np.asarray(ratio.reshape((int(args.surf_n_theta), int(args.surf_n_phi))))
    Kmag = np.asarray(jnp.linalg.norm(K, axis=-1))

    if not args.no_paraview:
        assert outdir is not None
        pv_dir = ensure_dir(Path(outdir) / "paraview")

        surf = write_vtu(
            pv_dir / "winding_surface_best.vtu",
            torus_surface_to_vtu(
                surface=surface,
                point_data={
                    "Phi": np.asarray(Phi_sv).reshape(-1),
                    "K": np.asarray(K).reshape(-1, 3),
                    "|K|": np.asarray(jnp.linalg.norm(K, axis=-1)).reshape(-1),
                },
            ),
        )
        tgt = write_vtu(
            pv_dir / "target_points.vtu",
            point_cloud_to_vtu(
                points=np.asarray(eval_points, dtype=float),
                point_data={
                    "Bn_over_B": np.asarray(ratio, dtype=float),
                    "n_hat": np.asarray(normals, dtype=float),
                    "weight": np.asarray(weights, dtype=float),
                },
            ),
        )
        scene = write_vtm(pv_dir / "scene.vtm", {"winding_surface_best": surf.name, "target_points": tgt.name})
        print(f"ParaView scene: {scene}")

    if args.no_plots:
        return

    # L-curve style plot.
    import matplotlib.pyplot as plt

    regK_arr = np.array([r["reg_K"] for r in rows], dtype=float)
    K_rms_arr = np.array([r["K_rms_A_per_m"] for r in rows], dtype=float)
    rms_arr = np.array([r["Bn_over_B_rms"] for r in rows], dtype=float)
    max_arr = np.array([r["Bn_over_B_max"] for r in rows], dtype=float)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.loglog(K_rms_arr, max_arr, "o-", label=r"max$|B_n/B|$")
    ax.loglog(K_rms_arr, rms_arr, "s--", label=r"rms$(B_n/B)$")
    for rk, x, y in zip(regK_arr, K_rms_arr, max_arr, strict=True):
        ax.annotate(f"{rk:.0e}", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel(r"$K_{\mathrm{rms}}$ [A/m]")
    ax.set_ylabel(r"$B_n/B$ [1]")
    ax.set_title("Regularization scan (tradeoff curve)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")
    savefig(fig, Path(outdir) / "l_curve.png", dpi=int(args.dpi))

    plot_surface_map(
        phi=np.asarray(target.phi),
        theta=np.asarray(target.theta),
        data=ratio_grid,
        title=f"Target map: Bn/B (best reg_K={regKs[-1]:.0e})",
        cbar_label=r"$B_n/B$ [1]",
        path=Path(outdir) / "maps" / "Bn_over_B_best.png",
        cmap="coolwarm",
        vmin=-float(max_abs),
        vmax=float(max_abs),
    )

    plot_surface_map(
        phi=np.asarray(surface.phi),
        theta=np.asarray(surface.theta),
        data=Kmag,
        title=r"Winding surface: $|K|$ (best)",
        cbar_label=r"$|K|$ [A/m]",
        path=Path(outdir) / "maps" / "Kmag_best.png",
        cmap="viridis",
    )

    # Summary text output for quick copy/paste into notes.
    print("Final (best reg_K) summary:")
    print(f"  reg_K={regKs[-1]:.3e}")
    print(f"  rms(Bn/B)={float(rms):.3e}")
    print(f"  max|Bn/B|={float(max_abs):.3e}")


if __name__ == "__main__":
    main()
