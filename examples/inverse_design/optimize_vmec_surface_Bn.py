#!/usr/bin/env python3
"""Example: optimize currents on a circular torus to reduce normalized (B·n)/|B| on a VMEC surface.

Goal:
  - Given a target "plasma" boundary surface from a VMEC input file, minimize the
    *normalized normal field* on that surface:

        L = ⟨ ((B_total · n_hat)/|B_total|)^2 ⟩_surface  +  regularization

Models:
  - `--model electrodes`: discrete injected currents on the torus surface + an imposed
    analytic toroidal background field ~ 1/R.
  - `--model current-potential`: REGCOIL-like current potential Φ on the torus surface
    with optional net poloidal/toroidal currents.

Important notes:
  - This is a *prototype* objective. Driving (B·n)/|B|≈0 is a necessary (but not sufficient)
    ingredient for a magnetic surface.
  - The VMEC boundary is shifted/scaled (if needed) so it fits inside the circular torus.

Run:
  python examples/inverse_design/optimize_vmec_surface_Bn.py --vmec-input examples/data/vmec/input.QA_nfp2
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
from pathlib import Path

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jaxopt import LBFGS

from torus_solver import make_torus_surface
from torus_solver.biot_savart import MU0, biot_savart_surface
from torus_solver.current_potential import surface_current_from_current_potential_with_net_currents
from torus_solver.fieldline import trace_field_lines_batch
from torus_solver.fields import ideal_toroidal_field, tokamak_like_field
from torus_solver.metrics import bn_over_B_metrics, weighted_p_norm
from torus_solver.optimize import SourceParams, enforce_net_zero, surface_solution
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
from torus_solver.targets import apply_fit_to_RZ_and_derivatives, vmec_target_surface
from torus_solver.vmec import read_vmec_boundary, vmec_boundary_RZ_and_derivatives
import torus_solver.plotting as tplot


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class PotentialParams:
    """Fourier coefficients for a single-valued current potential on the coil surface."""

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
        repo_root / vmec_input,  # allow paths relative to the repo root
        repo_root / "examples" / vmec_input,
        repo_root / "examples" / "data" / "vmec" / Path(vmec_input).name,
    ]
    for cand in candidates:
        if cand.exists():
            return cand

    msg = (
        "VMEC input file not found.\n"
        f"  got: {vmec_input}\n"
        f"  tried: {p.resolve()}\n"
        + "".join(f"  tried: {c}\n" for c in candidates)
        + "Tip: run with `--vmec-input examples/data/vmec/input.QA_nfp2` (from the repo root)."
    )
    raise FileNotFoundError(msg)


def _distance_to_target_surface_RZ(
    *,
    traj_xyz: np.ndarray,  # (n_steps+1, n_lines, 3)
    phi_grid: np.ndarray,  # (Nφ,)
    R_surf: np.ndarray,  # (Nθ,Nφ)
    Z_surf: np.ndarray,  # (Nθ,Nφ)
) -> np.ndarray:
    """Approximate distance-to-surface by nearest-φ slice and min distance in (R,Z)."""
    if traj_xyz.ndim != 3 or traj_xyz.shape[-1] != 3:
        raise ValueError(f"Expected traj_xyz shape (n_steps+1, n_lines, 3), got {traj_xyz.shape}")
    if R_surf.shape != Z_surf.shape:
        raise ValueError("R_surf and Z_surf must have the same shape.")
    if phi_grid.ndim != 1:
        raise ValueError("phi_grid must be 1D.")

    x = traj_xyz[..., 0]
    y = traj_xyz[..., 1]
    z = traj_xyz[..., 2]
    R = np.sqrt(x * x + y * y)
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    # Linear interpolation in φ (assumes uniform grid on [0,2π)).
    nphi = int(phi_grid.size)
    dphi = 2 * np.pi / nphi
    u = phi / dphi
    i0 = np.floor(u).astype(int) % nphi
    t = u - np.floor(u)
    i1 = (i0 + 1) % nphi

    flat0 = i0.reshape((-1,))
    flat1 = i1.reshape((-1,))
    t_flat = t.reshape((-1,))[None, :]

    R0 = R_surf[:, flat0]
    R1 = R_surf[:, flat1]
    Z0 = Z_surf[:, flat0]
    Z1 = Z_surf[:, flat1]
    R_curve = (1.0 - t_flat) * R0 + t_flat * R1  # (Nθ, P)
    Z_curve = (1.0 - t_flat) * Z0 + t_flat * Z1  # (Nθ, P)
    R_pt = R.reshape((-1,))[None, :]
    Z_pt = z.reshape((-1,))[None, :]
    dist2 = (R_curve - R_pt) ** 2 + (Z_curve - Z_pt) ** 2
    dist = np.sqrt(np.min(dist2, axis=0))
    return dist.reshape(i0.shape)


def _distance_to_target_surface_xyz(
    *,
    traj_xyz: np.ndarray,  # (n_steps+1, n_lines, 3)
    surface_xyz: np.ndarray,  # (Nθ,Nφ,3)
    chunk: int = 1024,
) -> np.ndarray:
    """Distance-to-surface using nearest neighbor over a discrete xyz point cloud.

    This is more robust than the (R,Z)-slice proxy for fully 3D surfaces.
    """
    if traj_xyz.ndim != 3 or traj_xyz.shape[-1] != 3:
        raise ValueError(f"Expected traj_xyz shape (n_steps+1, n_lines, 3), got {traj_xyz.shape}")
    if surface_xyz.ndim != 3 or surface_xyz.shape[-1] != 3:
        raise ValueError(f"Expected surface_xyz shape (Nθ,Nφ,3), got {surface_xyz.shape}")
    if chunk <= 0:
        raise ValueError("chunk must be positive.")

    surf = surface_xyz.reshape((-1, 3)).astype(np.float32)  # (M,3)
    surf2 = np.sum(surf * surf, axis=1, dtype=np.float32)[None, :]  # (1,M)

    pts = traj_xyz.reshape((-1, 3)).astype(np.float32)  # (P,3)
    P = pts.shape[0]
    out = np.empty((P,), dtype=np.float32)

    for i in range(0, P, chunk):
        t = pts[i : i + chunk]  # (C,3)
        t2 = np.sum(t * t, axis=1, dtype=np.float32)[:, None]  # (C,1)
        d2 = t2 + surf2 - 2.0 * (t @ surf.T)  # (C,M)
        out[i : i + t.shape[0]] = np.min(d2, axis=1)

    return np.sqrt(out, dtype=np.float32).reshape(traj_xyz.shape[:-1])


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        type=str,
        default="current-potential",
        choices=["current-potential", "electrodes"],
        help="Optimization model for currents on the circular torus winding surface.",
    )
    p.add_argument("--vmec-input", type=str, default="examples/data/vmec/input.QA_nfp2")

    # Winding surface (circular torus) geometry.
    p.add_argument("--R0", type=float, default=1.0, help="Torus major radius [m]")
    p.add_argument("--a", type=float, default=0.3, help="Torus minor radius [m]")
    p.add_argument("--n-theta", type=int, default=48, help="Torus grid Nθ")
    p.add_argument("--n-phi", type=int, default=48, help="Torus grid Nφ")

    # Target surface sampling.
    p.add_argument("--surf-n-theta", type=int, default=32, help="Target surface sampling Nθ")
    p.add_argument("--surf-n-phi", type=int, default=56, help="Target surface sampling Nφ")
    p.add_argument("--val-surf-n-theta", type=int, default=0, help="Validation sampling Nθ (0 disables)")
    p.add_argument("--val-surf-n-phi", type=int, default=0, help="Validation sampling Nφ (0 disables)")
    p.add_argument("--fit-margin", type=float, default=0.7, help="Require rho_max <= fit_margin * a")

    # Toroidal field scale [T]. In electrode mode this is an imposed background field.
    # In current-potential mode this sets net_poloidal_current via Ampere's law.
    p.add_argument("--B0", type=float, default=1.0, help="Toroidal field scale at R=R0 [T]")
    p.add_argument(
        "--Bpol0",
        type=float,
        default=0.0,
        help=(
            "Optional external poloidal background field scale at R=R0 [T]. "
            "This is added to the total field as (Bpol0*R0/R) e_theta (a tokamak-like proxy). "
            "Default 0 disables."
        ),
    )

    # Electrode model / optimization.
    p.add_argument("--n-sources", type=int, default=32)
    p.add_argument("--sigma-theta", type=float, default=0.25)
    p.add_argument("--sigma-phi", type=float, default=0.25)
    p.add_argument("--sigma-s", type=float, default=1.0, help="Surface conductivity scale (multiplies K)")
    p.add_argument(
        "--current-scale",
        type=float,
        default=None,
        help=(
            "Physical current scale [A] per unit of `currents_raw`. "
            "If omitted, uses an Ampere-law estimate ~ (2π R0 B0 / μ0)."
        ),
    )
    p.add_argument("--init-current-raw-rms", type=float, default=1.0, help="Init RMS of `currents_raw`")
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument(
        "--reg-currents",
        type=float,
        default=1e-3,
        help="Current regularization weight on ⟨(I/I_scale)^2⟩ (dimensionless).",
    )
    p.add_argument("--reg-positions", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cg-tol", type=float, default=1e-10)
    p.add_argument("--cg-maxiter", type=int, default=2000)
    p.add_argument("--use-preconditioner", action="store_true")

    # Current-potential model / optimization (REGCOIL-like).
    p.add_argument("--mpol-potential", type=int, default=23, help="Max poloidal mode number m")
    p.add_argument("--ntor-potential", type=int, default=11, help="Max toroidal mode number n")
    p.add_argument(
        "--net-poloidal-current",
        type=float,
        default=None,
        help="Net poloidal current [A]. If omitted, uses 2π R0 B0 / μ0.",
    )
    p.add_argument("--net-toroidal-current", type=float, default=0.0, help="Net toroidal current [A]")
    p.add_argument(
        "--phi-scale",
        type=float,
        default=None,
        help=(
            "Scale factor [A] multiplying the single-valued current potential Fourier coefficients. "
            "If omitted, uses max(|Ipol|,|Itor|,1)."
        ),
    )
    p.add_argument(
        "--reg-K",
        type=float,
        default=1e-7,
        help="Regularization weight on area-mean |K|^2 (dimensionless).",
    )
    p.add_argument(
        "--bn-p",
        type=float,
        default=8.0,
        help="Use a weighted p-norm of (Bn/|B|) in the objective (p=2 is RMS, larger p penalizes peaks).",
    )
    p.add_argument(
        "--bn-offset-m",
        type=float,
        default=0.0,
        help=(
            "Also penalize (Bn/|B|) on surfaces offset by ±bn_offset_m along the target surface normal. "
            "Set to 0 to disable."
        ),
    )
    p.add_argument("--lbfgs-tol", type=float, default=1e-9)

    # Fieldline plot controls.
    p.add_argument("--no-fieldlines", action="store_true")
    p.add_argument("--n-fieldlines", type=int, default=12)
    p.add_argument("--fieldline-steps", type=int, default=600)
    p.add_argument("--fieldline-ds", type=float, default=0.02)
    p.add_argument("--drift-surf-n-theta", type=int, default=0, help="Drift check surface grid Nθ (0 => 4*surf-n-theta)")
    p.add_argument("--drift-surf-n-phi", type=int, default=0, help="Drift check surface grid Nφ (0 => 4*surf-n-phi)")
    p.add_argument("--drift-stride", type=int, default=5, help="Subsample fieldline steps for drift metric")
    p.add_argument("--biot-savart-eps", type=float, default=1e-8)

    # Outputs.
    p.add_argument("--outdir", type=str, default="figures/optimize_vmec_surface_Bn")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-paraview", action="store_true", help="Disable ParaView (.vtu/.vtm) outputs")
    p.add_argument("--plot-stride", type=int, default=3)
    args = p.parse_args()

    outdir = None
    if (not args.no_plots) or (not args.no_paraview):
        outdir = ensure_dir(args.outdir)
        if not args.no_plots:
            set_plot_style()
            ensure_dir(outdir / "target")
            ensure_dir(outdir / "surface")
            ensure_dir(outdir / "optim")
            ensure_dir(outdir / "geometry")
            ensure_dir(outdir / "fieldlines")
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
    theta_s = target.theta
    phi_s = target.phi
    xyz = target.xyz
    n_hat = target.normals
    w_area = target.weights
    fit = target.fit

    shift_R = float(target.fit.shift_R)
    scale_rho = float(target.fit.scale_rho)
    rho_max = float(target.fit.rho_max_before_m)

    print("Fitting target surface into the circular torus")
    print(f"  torus: R0={args.R0} a={args.a}  fit_margin={args.fit_margin}")
    print(f"  shift_R = {shift_R:+.6e} m")
    print(f"  rho_max(before) = {rho_max:.6e} m   => scale={scale_rho:.6e}")
    print(f"  rho_max(after)  = {rho_max*scale_rho:.6e} m  (target <= {args.fit_margin*args.a:.6e} m)")

    eval_points = xyz.reshape((-1, 3))
    normals = n_hat.reshape((-1, 3))
    weights = w_area.reshape((-1,))

    print("Building winding surface (circular torus)")
    surface = make_torus_surface(R0=args.R0, a=args.a, n_theta=args.n_theta, n_phi=args.n_phi)

    if args.model == "current-potential":
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

        M = mpol + 1
        N = 2 * ntor + 1

        mask = jnp.ones((M, N), dtype=jnp.float64)
        mask = mask.at[0, ntor].set(0.0)  # exclude constant (m=0,n=0)

        th2 = surface.theta[None, None, :, None]
        ph2 = surface.phi[None, None, None, :]
        m_idx = jnp.arange(M, dtype=jnp.float64)[:, None, None, None]
        n_idx = (jnp.arange(-ntor, ntor + 1, dtype=jnp.float64) * float(nfp))[None, :, None, None]
        angle = m_idx * th2 - n_idx * ph2
        cosA = jnp.cos(angle)
        sinA = jnp.sin(angle)

        def Phi_from_params(pot: PotentialParams) -> jnp.ndarray:
            Phi_unit = jnp.sum(
                (pot.cos_mn * mask)[:, :, None, None] * cosA
                + (pot.sin_mn * mask)[:, :, None, None] * sinA,
                axis=(0, 1),
            )
            return phi_scale_A * Phi_unit

        # Regularization scaling: K ~ B0/mu0 if net_poloidal_current comes from B0.
        K_scale = 1.0
        if net_poloidal_current_A != 0.0:
            K_scale = max(K_scale, abs(net_poloidal_current_A) / (2 * np.pi * max(float(args.R0), 1e-12)))
        if net_toroidal_current_A != 0.0:
            K_scale = max(K_scale, abs(net_toroidal_current_A) / (2 * np.pi * max(float(args.a), 1e-12)))

        # Objective sampling: optionally include a "buffer" of offset surfaces to encourage
        # tangency in a neighborhood of the target surface (helps reduce field-line drift).
        n_target = int(eval_points.shape[0])
        bn_offset_m = float(args.bn_offset_m)
        if bn_offset_m < 0.0:
            raise ValueError("--bn-offset-m must be >= 0.")
        if bn_offset_m == 0.0:
            eval_obj = eval_points
            normals_obj = normals
            weights_obj = weights
            n_layers = 1
        else:
            layers = jnp.array([0.0, bn_offset_m, -bn_offset_m], dtype=jnp.float64)
            eval_obj = jnp.concatenate([eval_points + off * normals for off in layers], axis=0)
            normals_obj = jnp.concatenate([normals for _ in range(int(layers.size))], axis=0)
            weights_obj = jnp.concatenate([weights for _ in range(int(layers.size))], axis=0)
            n_layers = int(layers.size)

        def loss_fn_potential(pot: PotentialParams):
            Phi_sv = Phi_from_params(pot)
            K = surface_current_from_current_potential_with_net_currents(
                surface,
                Phi_sv,
                net_poloidal_current_A=net_poloidal_current_A,
                net_toroidal_current_A=net_toroidal_current_A,
            )
            B_obj = biot_savart_surface(surface, K, eval_obj, eps=float(args.biot_savart_eps))
            if float(args.Bpol0) != 0.0:
                # Optional "plasma current" proxy: add an external poloidal component ~ 1/R.
                B_obj = B_obj + tokamak_like_field(
                    eval_obj, B_tor0=0.0, B_pol0=float(args.Bpol0), R0=float(args.R0)
                )
            ratio_obj = jnp.sum(B_obj * normals_obj, axis=-1) / (jnp.linalg.norm(B_obj, axis=-1) + 1e-30)
            bn_metric = weighted_p_norm(ratio_obj, weights_obj, p=float(args.bn_p))
            loss_bn = bn_metric * bn_metric

            # Report metrics on the central (unshifted) target surface only.
            B0 = B_obj[:n_target, :]
            ratio, rms, max_abs = bn_over_B_metrics(B0, normals, weights)

            K2 = jnp.sum(K * K, axis=-1)
            mean_K2 = jnp.sum(K2 * surface.area_weights) / (jnp.sum(surface.area_weights) + 1e-30)
            loss_reg = float(args.reg_K) * (mean_K2 / (K_scale * K_scale))
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

        init_pot = PotentialParams(
            cos_mn=jnp.zeros((M, N), dtype=jnp.float64),
            sin_mn=jnp.zeros((M, N), dtype=jnp.float64),
        )

        print("Optimization setup (current potential / REGCOIL-like)")
        print(f"  potential modes: mpol={mpol}  ntor={ntor}  nfp={nfp}")
        print(f"  net currents:    Ipol={net_poloidal_current_A:+.3e} A  Itor={net_toroidal_current_A:+.3e} A")
        print(f"  Phi_scale:       {phi_scale_A:.3e} A (coefficients are dimensionless)")
        print(f"  objective:       minimize weighted p-norm (p={args.bn_p:g}) of (Bn/|B|) on target surface")
        print(f"  reg:             reg_K={args.reg_K:.3e}  (K_scale={K_scale:.3e} A/m)")
        print(f"  solver:          L-BFGS maxiter={args.n_steps} tol={args.lbfgs_tol}")

        loss0, aux0 = loss_fn_potential(init_pot)
        print(
            "init: loss={:.3e} bn={:.3e} reg={:.3e}  rms(Bn/B)={:.3e}  max|Bn/B|={:.3e}  K_rms={:.3e} A/m".format(
                float(aux0["loss"]),
                float(aux0["loss_bn"]),
                float(aux0["loss_reg"]),
                float(aux0["Bn_over_B_rms"]),
                float(aux0["Bn_over_B_max"]),
                float(aux0["K_rms"]),
            )
        )
        print(f"      p-norm(Bn/B)={float(aux0['Bn_over_B_p']):.3e}")
        g0 = jax.grad(lambda p: loss_fn_potential(p)[0])(init_pot)
        g0_cos = float(jnp.linalg.norm(g0.cos_mn))
        g0_sin = float(jnp.linalg.norm(g0.sin_mn))
        print(f"      grad norms: ||dL/dcos||={g0_cos:.3e}  ||dL/dsin||={g0_sin:.3e}")

        solver = LBFGS(fun=loss_fn_potential, has_aux=True, maxiter=int(args.n_steps), tol=float(args.lbfgs_tol))
        target_max = 5e-3  # 0.5% (matches README/docs target)
        t0 = time.perf_counter()
        pot = init_pot
        state = solver.init_state(pot)
        for k in range(int(args.n_steps)):
            pot, state = solver.update(pot, state)
            aux = state.aux
            if (k % 25) == 0 or k == int(args.n_steps) - 1:
                print(
                    "iter={:4d} loss={:.3e}  rms(Bn/B)={:.3e}  max|Bn/B|={:.3e}  K_rms={:.3e} A/m".format(
                        k,
                        float(aux["loss"]),
                        float(aux["Bn_over_B_rms"]),
                        float(aux["Bn_over_B_max"]),
                        float(aux["K_rms"]),
                    )
                )
            if float(aux["Bn_over_B_max"]) < target_max:
                print(f"Early stop: reached max|Bn/B|<{target_max:g} at iter={k}.")
                break
        pot.cos_mn.block_until_ready()
        t1 = time.perf_counter()
        pot_best = pot
        try:
            print(
                "LBFGS state: iter={}  error={:.3e}  stepsize={:.3e}  failed_linesearch={}".format(
                    int(state.iter_num),
                    float(state.error),
                    float(state.stepsize),
                    bool(state.failed_linesearch),
                )
            )
        except Exception as e:
            print("LBFGS state: <unavailable>", repr(e))

        loss1, aux1 = loss_fn_potential(pot_best)
        print(
            "final: loss={:.3e} bn={:.3e} reg={:.3e}  rms(Bn/B)={:.3e}  max|Bn/B|={:.3e}  K_rms={:.3e} A/m".format(
                float(aux1["loss"]),
                float(aux1["loss_bn"]),
                float(aux1["loss_reg"]),
                float(aux1["Bn_over_B_rms"]),
                float(aux1["Bn_over_B_max"]),
                float(aux1["K_rms"]),
            )
        )
        print(f"       p-norm(Bn/B)={float(aux1['Bn_over_B_p']):.3e}")
        print(f"Wall time: {t1 - t0:.2f} s")

        # Compute init/final fields for maps and tracing.
        Phi0_sv = Phi_from_params(init_pot)
        K0 = surface_current_from_current_potential_with_net_currents(
            surface,
            Phi0_sv,
            net_poloidal_current_A=net_poloidal_current_A,
            net_toroidal_current_A=net_toroidal_current_A,
        )
        B0_tot = biot_savart_surface(surface, K0, eval_points, eps=float(args.biot_savart_eps))
        if float(args.Bpol0) != 0.0:
            B0_tot = B0_tot + tokamak_like_field(
                eval_points, B_tor0=0.0, B_pol0=float(args.Bpol0), R0=float(args.R0)
            )
        ratio0, rms0, max0 = bn_over_B_metrics(B0_tot, normals, weights)

        Phi1_sv = Phi_from_params(pot_best)
        K1 = surface_current_from_current_potential_with_net_currents(
            surface,
            Phi1_sv,
            net_poloidal_current_A=net_poloidal_current_A,
            net_toroidal_current_A=net_toroidal_current_A,
        )
        B1_tot = biot_savart_surface(surface, K1, eval_points, eps=float(args.biot_savart_eps))
        if float(args.Bpol0) != 0.0:
            B1_tot = B1_tot + tokamak_like_field(
                eval_points, B_tor0=0.0, B_pol0=float(args.Bpol0), R0=float(args.R0)
            )
        ratio1, rms1, max1 = bn_over_B_metrics(B1_tot, normals, weights)

        print("Bn/B summary on target surface (area-weighted)")
        print(f"  rms(Bn/B) init  = {float(rms0):.3e}")
        print(f"  max|Bn/B| init  = {float(max0):.3e}")
        print(f"  rms(Bn/B) final = {float(rms1):.3e}")
        print(f"  max|Bn/B| final = {float(max1):.3e}")
        print("Target threshold: max|Bn/B| < 5e-3 (0.5%)")
        if float(max1) > 5e-3:
            print("Tip: increase --n-steps (e.g. 200+) and/or run the regularization scan to pick reg_K.")

        if int(args.val_surf_n_theta) > 0 and int(args.val_surf_n_phi) > 0:
            print(
                f"Validation on a finer target grid: Nθ={int(args.val_surf_n_theta)} Nφ={int(args.val_surf_n_phi)}"
            )
            theta_v = jnp.linspace(0.0, 2 * jnp.pi, int(args.val_surf_n_theta), endpoint=False, dtype=jnp.float64)
            phi_v = jnp.linspace(0.0, 2 * jnp.pi, int(args.val_surf_n_phi), endpoint=False, dtype=jnp.float64)
            Rv, Zv, Rth_v, Rph_v, Zth_v, Zph_v = vmec_boundary_RZ_and_derivatives(boundary, theta=theta_v, phi=phi_v)
            Rv_fit, Zv_fit, Rth_v_fit, Rph_v_fit, Zth_v_fit, Zph_v_fit = apply_fit_to_RZ_and_derivatives(
                R=Rv,
                Z=Zv,
                R_theta=Rth_v,
                R_phi=Rph_v,
                Z_theta=Zth_v,
                Z_phi=Zph_v,
                torus_R0=float(args.R0),
                fit=fit,
            )

            phv2 = phi_v[None, :]
            cv = jnp.cos(phv2)
            sv = jnp.sin(phv2)
            xv = Rv_fit * cv
            yv = Rv_fit * sv
            zv = Zv_fit
            xyz_v = jnp.stack([xv, yv, zv], axis=-1)

            rth_v_xyz = jnp.stack([Rth_v_fit * cv, Rth_v_fit * sv, Zth_v_fit], axis=-1)
            rph_v_xyz = jnp.stack(
                [Rph_v_fit * cv - Rv_fit * sv, Rph_v_fit * sv + Rv_fit * cv, Zph_v_fit], axis=-1
            )
            nvec_v = jnp.cross(rth_v_xyz, rph_v_xyz)
            nnorm_v = jnp.linalg.norm(nvec_v, axis=-1)
            nhat_v = nvec_v / (nnorm_v[..., None] + 1e-30)

            eval_v = xyz_v.reshape((-1, 3))
            normals_v = nhat_v.reshape((-1, 3))
            weights_v = nnorm_v.reshape((-1,))

            Bv_tot = biot_savart_surface(surface, K1, eval_v, eps=float(args.biot_savart_eps))
            _ratio_v, rms_v, max_v = bn_over_B_metrics(Bv_tot, normals_v, weights_v)
            print(f"  validation rms(Bn/B)  = {float(rms_v):.3e}")
            print(f"  validation max|Bn/B|  = {float(max_v):.3e}")

        # Field lines of the vacuum field from the optimized winding-surface current.
        traj_np: np.ndarray | None = None
        dist: np.ndarray | None = None
        if not args.no_fieldlines:
            print("Tracing field lines for final field (current-potential on winding surface)...")
            # Seed on the target surface at phi≈0.
            nline = int(args.n_fieldlines)
            theta_seed_idx = np.linspace(0, int(args.surf_n_theta) - 1, nline, endpoint=False).astype(int)
            seeds = xyz.reshape((int(args.surf_n_theta), int(args.surf_n_phi), 3))[theta_seed_idx, 0, :]
            ratio1_grid = np.asarray(ratio1.reshape((int(args.surf_n_theta), int(args.surf_n_phi))))
            seed_ratio = ratio1_grid[theta_seed_idx, 0]
            print(
                "  seeds: max|Bn/B|={:.3e}  mean|Bn/B|={:.3e}".format(
                    float(np.max(np.abs(seed_ratio))),
                    float(np.mean(np.abs(seed_ratio))),
                )
            )
            print("  seeds |Bn/B| per-line:", ", ".join(f"{i}:{abs(seed_ratio[i]):.3e}" for i in range(seed_ratio.size)))
            nhat_grid = np.asarray(n_hat.reshape((int(args.surf_n_theta), int(args.surf_n_phi), 3)))
            seed_normals = nhat_grid[theta_seed_idx, 0, :]
            B_seed = np.asarray(
                biot_savart_surface(surface, K1, jnp.asarray(seeds, dtype=jnp.float64), eps=float(args.biot_savart_eps))
            )
            B_seed_mag = np.linalg.norm(B_seed, axis=-1, keepdims=True) + 1e-30
            print(
                "  seeds: |B| min={:.3e} T  max={:.3e} T".format(
                    float(np.min(B_seed_mag)),
                    float(np.max(B_seed_mag)),
                )
            )
            ratio_seed_direct = np.sum((B_seed / B_seed_mag) * seed_normals, axis=-1)
            print(
                "  seeds (direct): max|Bn/B|={:.3e}  mean|Bn/B|={:.3e}".format(
                    float(np.max(np.abs(ratio_seed_direct))),
                    float(np.mean(np.abs(ratio_seed_direct))),
                )
            )

            def B_fn(xyz_pts: jnp.ndarray) -> jnp.ndarray:
                return biot_savart_surface(surface, K1, xyz_pts, eps=float(args.biot_savart_eps))

            traj = jax.jit(
                lambda x0: trace_field_lines_batch(
                    B_fn,
                    jnp.asarray(x0, dtype=jnp.float64),
                    step_size=float(args.fieldline_ds),
                    n_steps=int(args.fieldline_steps),
                    normalize=True,
                )
            )(seeds)
            traj.block_until_ready()
            traj_np = np.asarray(traj)

            # Drift metric against a *denser* sampling of the target surface, to reduce
            # discretization error when field lines move tangentially on the surface.
            drift_stride = max(1, int(args.drift_stride))
            if traj_np.shape[0] <= drift_stride:
                drift_stride = 1
            drift_n_theta = (
                int(args.drift_surf_n_theta)
                if int(args.drift_surf_n_theta) > 0
                else min(256, 4 * int(args.surf_n_theta))
            )
            drift_n_phi = (
                int(args.drift_surf_n_phi) if int(args.drift_surf_n_phi) > 0 else min(512, 4 * int(args.surf_n_phi))
            )
            theta_d = jnp.linspace(0.0, 2 * jnp.pi, drift_n_theta, endpoint=False, dtype=jnp.float64)
            phi_d = jnp.linspace(0.0, 2 * jnp.pi, drift_n_phi, endpoint=False, dtype=jnp.float64)
            Rd, Zd, Rd_th, Rd_ph, Zd_th, Zd_ph = vmec_boundary_RZ_and_derivatives(boundary, theta=theta_d, phi=phi_d)
            Rd_fit, Zd_fit, Rd_th_fit, Rd_ph_fit, Zd_th_fit, Zd_ph_fit = apply_fit_to_RZ_and_derivatives(
                R=Rd,
                Z=Zd,
                R_theta=Rd_th,
                R_phi=Rd_ph,
                Z_theta=Zd_th,
                Z_phi=Zd_ph,
                torus_R0=float(args.R0),
                fit=fit,
            )
            ph_d2 = phi_d[None, :]
            cd = jnp.cos(ph_d2)
            sd = jnp.sin(ph_d2)
            xyz_d = jnp.stack([Rd_fit * cd, Rd_fit * sd, Zd_fit], axis=-1)

            traj_sub = traj_np[::drift_stride, :, :]
            dist = _distance_to_target_surface_xyz(traj_xyz=traj_sub, surface_xyz=np.asarray(xyz_d), chunk=256)
            dist_max = float(np.max(dist))
            dist_mean = float(np.mean(dist))
            dist0 = float(np.max(dist[0]))
            dist_end = float(np.max(dist[-1]))
            print(
                "Field-line drift (approx, drift_stride={}): mean distance={:.3e} m  max distance={:.3e} m  "
                "max(d0)={:.3e} m  max(d_end)={:.3e} m".format(drift_stride, dist_mean, dist_max, dist0, dist_end)
            )
            end_per_line = dist[-1]
            worst = np.argsort(end_per_line)[-min(5, end_per_line.size) :][::-1]
            print("  drift end per-line (worst):", ", ".join(f"{i}:{end_per_line[i]:.3e}m" for i in worst))

        if not args.no_paraview:
            assert outdir is not None
            pv_dir = ensure_dir(outdir / "paraview")

            Kmag0 = np.asarray(jnp.linalg.norm(K0, axis=-1)).reshape(-1)
            Kmag1 = np.asarray(jnp.linalg.norm(K1, axis=-1)).reshape(-1)
            surf_init = write_vtu(
                pv_dir / "winding_surface_init.vtu",
                torus_surface_to_vtu(
                    surface=surface,
                    point_data={
                        "Phi": np.asarray(Phi0_sv).reshape(-1),
                        "K": np.asarray(K0).reshape(-1, 3),
                        "|K|": Kmag0,
                    },
                ),
            )
            surf_final = write_vtu(
                pv_dir / "winding_surface_final.vtu",
                torus_surface_to_vtu(
                    surface=surface,
                    point_data={
                        "Phi": np.asarray(Phi1_sv).reshape(-1),
                        "K": np.asarray(K1).reshape(-1, 3),
                        "|K|": Kmag1,
                    },
                ),
            )

            tgt = write_vtu(
                pv_dir / "target_points.vtu",
                point_cloud_to_vtu(
                    points=np.asarray(eval_points, dtype=float),
                    point_data={
                        "Bn_over_B_init": np.asarray(ratio0, dtype=float),
                        "Bn_over_B_final": np.asarray(ratio1, dtype=float),
                        "n_hat": np.asarray(normals, dtype=float),
                        "weight": np.asarray(weights, dtype=float),
                    },
                ),
            )

            blocks: dict[str, str] = {
                "winding_surface_init": surf_init.name,
                "winding_surface_final": surf_final.name,
                "target_points": tgt.name,
            }
            if traj_np is not None:
                traj_pv = np.transpose(traj_np, (1, 0, 2))
                fl = write_vtu(pv_dir / "fieldlines_final.vtu", fieldlines_to_vtu(traj=traj_pv))
                blocks["fieldlines_final"] = fl.name

            scene = write_vtm(pv_dir / "scene.vtm", blocks)
            print(f"ParaView scene: {scene}")

        if args.no_plots:
            return

        outdir = ensure_dir(args.outdir)

        ratio0_grid = np.asarray(ratio0.reshape((int(args.surf_n_theta), int(args.surf_n_phi))))
        ratio1_grid = np.asarray(ratio1.reshape((int(args.surf_n_theta), int(args.surf_n_phi))))
        plot_surface_map(
            phi=np.asarray(phi_s),
            theta=np.asarray(theta_s),
            data=ratio0_grid,
            title=r"Target surface $(B\cdot \hat n)/|B|$ (init)",
            cbar_label=r"$(B\cdot \hat n)/|B|$ [1]",
            cmap="coolwarm",
            path=outdir / "target" / "Bn_over_B_init.png",
        )
        plot_surface_map(
            phi=np.asarray(phi_s),
            theta=np.asarray(theta_s),
            data=ratio1_grid,
            title=r"Target surface $(B\cdot \hat n)/|B|$ (final)",
            cbar_label=r"$(B\cdot \hat n)/|B|$ [1]",
            cmap="coolwarm",
            path=outdir / "target" / "Bn_over_B_final.png",
        )

        # 3D geometry: torus + target surface point cloud.
        plot_3d_torus(
            torus_xyz=np.asarray(surface.r),
            electrodes_xyz=None,
            curve_xyz=None,
            eval_xyz=np.asarray(eval_points),
            stride=args.plot_stride,
            title="Torus + target surface points",
            path=outdir / "geometry" / "geometry.png",
        )

        if traj_np is not None:
            plot_fieldlines_3d(
                torus_xyz=np.asarray(surface.r),
                traj=traj_np,
                stride=args.plot_stride,
                line_stride=1,
                title="Final field lines (current-potential winding surface)",
                path=outdir / "fieldlines" / "fieldlines_final.png",
            )
        if dist is not None:
            fig, ax = tplot.plt.subplots(constrained_layout=True)
            drift_stride = max(1, int(args.drift_stride))
            step = np.arange(dist.shape[0]) * float(args.fieldline_ds) * float(drift_stride)
            ax.plot(step, dist.max(axis=1), label="max over lines")
            ax.plot(step, dist.mean(axis=1), label="mean over lines")
            ax.set_xlabel("Arc-length parameter [m] (approx, subsampled)")
            ax.set_ylabel("Distance to target surface [m]")
            ax.set_title("Field-line drift relative to target surface (nearest xyz point)")
            ax.legend()
            savefig(fig, outdir / "fieldlines" / "fieldline_drift.png", dpi=args.dpi)

        return

    # ---- electrode model below ----

    # Precompute background field at the evaluation points (for electrode model).
    B_bg = tokamak_like_field(eval_points, B_tor0=float(args.B0), B_pol0=float(args.Bpol0), R0=float(args.R0))
    ratio_bg, ratio_bg_rms, ratio_bg_max = bn_over_B_metrics(B_bg, normals, weights)

    if args.current_scale is None:
        # Ampere's law scaling: toroidal 1/R field at R=R0 from a poloidal current.
        mu0 = float(4e-7 * np.pi)
        if float(args.B0) == 0.0:
            args.current_scale = 1.0
        else:
            args.current_scale = float(2 * np.pi * args.R0 * abs(args.B0) / mu0)

    # Initialize electrode parameters.
    key = jax.random.key(int(args.seed))
    theta0 = 2 * jnp.pi * jax.random.uniform(key, (args.n_sources,))
    phi0 = 2 * jnp.pi * jax.random.uniform(jax.random.fold_in(key, 1), (args.n_sources,))
    I0 = float(args.init_current_raw_rms) * jax.random.normal(jax.random.fold_in(key, 2), (args.n_sources,))
    init = SourceParams(theta_src=theta0, phi_src=phi0, currents_raw=I0)

    print("Optimization setup")
    print(f"  electrodes: n={args.n_sources}  sigma_theta={args.sigma_theta} sigma_phi={args.sigma_phi}")
    print(f"  objective:  minimize area-weighted <(Bn/|B|)^2> on target surface")
    print(f"  background: tokamak-like 1/R field  Btor0={args.B0}T  Bpol0={args.Bpol0}T at R0={args.R0}")
    print(f"  current_scale: {args.current_scale:.6e} A per unit(currents_raw)")
    print(f"  optim:      steps={args.n_steps} lr={args.lr} regI={args.reg_currents} regPos={args.reg_positions}")
    print(f"  cg:         tol={args.cg_tol} maxiter={args.cg_maxiter} preconditioner={bool(args.use_preconditioner)}")

    I_scale = float(args.current_scale) if float(args.current_scale) != 0.0 else 1.0

    def loss_fn(p_: SourceParams):
        # Forward solve on the winding surface.
        currents, s_src, V, K = surface_solution(
            surface,
            p_,
            sigma_theta=float(args.sigma_theta),
            sigma_phi=float(args.sigma_phi),
            sigma_s=float(args.sigma_s),
            current_scale=I_scale,
            tol=float(args.cg_tol),
            maxiter=int(args.cg_maxiter),
            use_preconditioner=bool(args.use_preconditioner),
        )
        B_shell = biot_savart_surface(surface, K, eval_points, eps=float(args.biot_savart_eps))
        B_tot = B_bg + B_shell
        _ratio, rms, max_abs = bn_over_B_metrics(B_tot, normals, weights)
        loss_bn = rms * rms

        # Dimensionless regularizer: ⟨(I/I_scale)^2⟩.
        loss_reg = float(args.reg_currents) * jnp.mean((currents / I_scale) ** 2)
        if float(args.reg_positions) != 0.0:
            loss_reg = loss_reg + float(args.reg_positions) * (
                jnp.mean(p_.theta_src * p_.theta_src) + jnp.mean(p_.phi_src * p_.phi_src)
            )

        loss = loss_bn + loss_reg
        aux = {
            "loss": loss,
            "loss_bn": loss_bn,
            "loss_reg": loss_reg,
            "Bn_over_B_rms": rms,
            "Bn_over_B_max": max_abs,
            "I_rms_A": jnp.sqrt(jnp.mean(currents * currents)),
        }
        return loss, aux

    opt = optax.adam(float(args.lr))
    opt_state = opt.init(init)

    @jax.jit
    def step(p_, s_):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p_)
        updates, s2 = opt.update(g, s_, p_)
        p2 = optax.apply_updates(p_, updates)
        return p2, s2, aux

    params = init
    state = opt_state
    history = {
        "loss": [],
        "loss_bn": [],
        "loss_reg": [],
        "Bn_over_B_rms": [],
        "Bn_over_B_max": [],
        "I_rms_A": [],
    }

    print("Initial diagnostics on target surface (area-weighted)")
    print(f"  background: rms(Bn/B)={float(ratio_bg_rms):.3e}  max|Bn/B|={float(ratio_bg_max):.3e}")
    currents_init, _s0, _V0, K_init = surface_solution(
        surface,
        init,
        sigma_theta=float(args.sigma_theta),
        sigma_phi=float(args.sigma_phi),
        sigma_s=float(args.sigma_s),
        current_scale=I_scale,
        tol=float(args.cg_tol),
        maxiter=int(args.cg_maxiter),
        use_preconditioner=bool(args.use_preconditioner),
    )
    B_shell_init = biot_savart_surface(surface, K_init, eval_points, eps=float(args.biot_savart_eps))
    B_tot_init = B_bg + B_shell_init
    _ratio_init, rms_init, max_init = bn_over_B_metrics(B_tot_init, normals, weights)
    print(f"  init total:  rms(Bn/B)={float(rms_init):.3e}  max|Bn/B|={float(max_init):.3e}")

    t0 = time.perf_counter()
    for k in range(int(args.n_steps)):
        params, state, aux = step(params, state)
        if k % 25 == 0 or k == int(args.n_steps) - 1:
            print(
                "step={:4d} loss={:.6e} bn={:.3e} reg={:.3e}  rms(Bn/B)={:.3e}  max|Bn/B|={:.3e}  I_rms={:.3e}A".format(
                    k,
                    float(aux["loss"]),
                    float(aux["loss_bn"]),
                    float(aux["loss_reg"]),
                    float(aux["Bn_over_B_rms"]),
                    float(aux["Bn_over_B_max"]),
                    float(aux["I_rms_A"]),
                )
            )
        for name in history:
            history[name].append(float(aux[name]))
    t1 = time.perf_counter()
    print(f"Wall time: {t1 - t0:.2f} s")
    print(f"Net current (projected): {float(jnp.sum(enforce_net_zero(params.currents_raw))):+.3e} A")

    # Compute initial/final Bn maps for plotting and summary.
    currents0, s0, V0, K0 = surface_solution(
        surface,
        init,
        sigma_theta=float(args.sigma_theta),
        sigma_phi=float(args.sigma_phi),
        sigma_s=float(args.sigma_s),
        current_scale=I_scale,
        tol=float(args.cg_tol),
        maxiter=int(args.cg_maxiter),
        use_preconditioner=bool(args.use_preconditioner),
    )
    currents1, s1, V1, K1 = surface_solution(
        surface,
        params,
        sigma_theta=float(args.sigma_theta),
        sigma_phi=float(args.sigma_phi),
        sigma_s=float(args.sigma_s),
        current_scale=I_scale,
        tol=float(args.cg_tol),
        maxiter=int(args.cg_maxiter),
        use_preconditioner=bool(args.use_preconditioner),
    )

    def B_total_from_K(K, pts):
        return tokamak_like_field(
            pts, B_tor0=float(args.B0), B_pol0=float(args.Bpol0), R0=float(args.R0)
        ) + biot_savart_surface(surface, K, pts, eps=float(args.biot_savart_eps))

    B0_tot = B_total_from_K(K0, eval_points)
    B1_tot = B_total_from_K(K1, eval_points)
    ratio0, rms0, max0 = bn_over_B_metrics(B0_tot, normals, weights)
    ratio1, rms1, max1 = bn_over_B_metrics(B1_tot, normals, weights)

    print("Bn/B summary on target surface (area-weighted)")
    print(f"  rms(Bn/B) init  = {float(rms0):.3e}")
    print(f"  max|Bn/B| init  = {float(max0):.3e}")
    print(f"  rms(Bn/B) final = {float(rms1):.3e}")
    print(f"  max|Bn/B| final = {float(max1):.3e}")

    traj_np = None
    if (not args.no_fieldlines) and ((not args.no_plots) or (not args.no_paraview)):
        print("Tracing field lines for final total field (background + Biot–Savart)...")
        theta_seed = jnp.linspace(0.0, 2 * jnp.pi, int(args.n_fieldlines), endpoint=False)
        rho = 0.5 * surface.a
        R_seed = surface.R0 + rho * jnp.cos(theta_seed)
        Z_seed = rho * jnp.sin(theta_seed)
        seeds = jnp.stack([R_seed, jnp.zeros_like(R_seed), Z_seed], axis=-1)

        def B_fn(xyz_pts: jnp.ndarray) -> jnp.ndarray:
            return tokamak_like_field(
                xyz_pts, B_tor0=float(args.B0), B_pol0=float(args.Bpol0), R0=float(args.R0)
            ) + biot_savart_surface(surface, K1, xyz_pts, eps=float(args.biot_savart_eps))

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

        Kmag0 = np.asarray(jnp.linalg.norm(K0, axis=-1)).reshape(-1)
        Kmag1 = np.asarray(jnp.linalg.norm(K1, axis=-1)).reshape(-1)
        surf_init = write_vtu(
            pv_dir / "winding_surface_init.vtu",
            torus_surface_to_vtu(
                surface=surface,
                point_data={
                    "V": np.asarray(V0).reshape(-1),
                    "s": np.asarray(s0).reshape(-1),
                    "K": np.asarray(K0).reshape(-1, 3),
                    "|K|": Kmag0,
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
                    "|K|": Kmag1,
                },
            ),
        )

        def torus_xyz(theta, phi):
            R = args.R0 + args.a * np.cos(theta)
            return np.stack([R * np.cos(phi), R * np.sin(phi), args.a * np.sin(theta)], axis=-1)

        el0_xyz = torus_xyz(np.asarray(init.theta_src), np.asarray(init.phi_src))
        el1_xyz = torus_xyz(np.asarray(params.theta_src), np.asarray(params.phi_src))
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
                    "I_raw": np.asarray(params.currents_raw, dtype=float),
                    "sign_I": np.sign(np.asarray(currents1, dtype=float)),
                },
            ),
        )

        tgt = write_vtu(
            pv_dir / "target_points.vtu",
            point_cloud_to_vtu(
                points=np.asarray(eval_points, dtype=float),
                point_data={
                    "Bn_over_B_init": np.asarray(ratio0, dtype=float),
                    "Bn_over_B_final": np.asarray(ratio1, dtype=float),
                    "n_hat": np.asarray(normals, dtype=float),
                    "weight": np.asarray(weights, dtype=float),
                },
            ),
        )

        blocks: dict[str, str] = {
            "winding_surface_init": surf_init.name,
            "winding_surface_final": surf_final.name,
            "electrodes_init": el_init.name,
            "electrodes_final": el_final.name,
            "target_points": tgt.name,
        }
        if traj_np is not None:
            traj_pv = np.transpose(traj_np, (1, 0, 2))
            fl = write_vtu(pv_dir / "fieldlines_final.vtu", fieldlines_to_vtu(traj=traj_pv))
            blocks["fieldlines_final"] = fl.name

        scene = write_vtm(pv_dir / "scene.vtm", blocks)
        print(f"ParaView scene: {scene}")

    if args.no_plots:
        return

    outdir = ensure_dir(args.outdir)

    # Target surface maps.
    ratio0_grid = np.asarray(ratio0.reshape((int(args.surf_n_theta), int(args.surf_n_phi))))
    ratio1_grid = np.asarray(ratio1.reshape((int(args.surf_n_theta), int(args.surf_n_phi))))
    plot_surface_map(
        phi=np.asarray(phi_s),
        theta=np.asarray(theta_s),
        data=ratio0_grid,
        title=r"Target surface $(B\cdot \hat n)/|B|$ (init)",
        cbar_label=r"$(B\cdot \hat n)/|B|$ [1]",
        cmap="coolwarm",
        path=outdir / "target" / "Bn_over_B_init.png",
    )
    plot_surface_map(
        phi=np.asarray(phi_s),
        theta=np.asarray(theta_s),
        data=ratio1_grid,
        title=r"Target surface $(B\cdot \hat n)/|B|$ (final)",
        cbar_label=r"$(B\cdot \hat n)/|B|$ [1]",
        cmap="coolwarm",
        path=outdir / "target" / "Bn_over_B_final.png",
    )

    # Optimization history.
    steps = list(range(len(history["loss"])))
    plot_loss_history(
        steps=steps,
        metrics={"loss": history["loss"], "loss_bn": history["loss_bn"], "loss_reg": history["loss_reg"]},
        title="Optimization history",
        path=outdir / "optim" / "loss_history.png",
    )
    plot_loss_history(
        steps=steps,
        metrics={
            "rms(Bn/B) [1]": history["Bn_over_B_rms"],
            "max|Bn/B| [1]": history["Bn_over_B_max"],
            "I_rms [A]": history["I_rms_A"],
        },
        title="rms/max(Bn/B) and I_rms",
        path=outdir / "optim" / "Bn_over_B_I_rms.png",
        yscale="log",
    )

    # 3D geometry: torus + target surface point cloud + electrodes.
    def torus_xyz(theta, phi):
        R = args.R0 + args.a * np.cos(theta)
        return np.stack([R * np.cos(phi), R * np.sin(phi), args.a * np.sin(theta)], axis=-1)

    electrodes0 = torus_xyz(np.asarray(init.theta_src), np.asarray(init.phi_src))
    electrodes1 = torus_xyz(np.asarray(params.theta_src), np.asarray(params.phi_src))

    plot_3d_torus(
        torus_xyz=np.asarray(surface.r),
        electrodes_xyz=electrodes0,
        curve_xyz=None,
        eval_xyz=np.asarray(eval_points),
        stride=args.plot_stride,
        title="Torus + target surface points + initial electrodes",
        path=outdir / "geometry" / "geometry_init.png",
    )
    plot_3d_torus(
        torus_xyz=np.asarray(surface.r),
        electrodes_xyz=electrodes1,
        curve_xyz=None,
        eval_xyz=np.asarray(eval_points),
        stride=args.plot_stride,
        title="Torus + target surface points + final electrodes",
        path=outdir / "geometry" / "geometry_final.png",
    )

    # Field lines of the *total* field.
    if traj_np is not None:
        plot_fieldlines_3d(
            torus_xyz=np.asarray(surface.r),
            traj=traj_np,
            stride=args.plot_stride,
            line_stride=1,
            title="Final total-field lines (background toroidal + optimized shell currents)",
            path=outdir / "fieldlines" / "fieldlines_final.png",
        )

    # Currents bar plot.
    fig, ax = tplot.plt.subplots(constrained_layout=True)
    idx = np.arange(int(args.n_sources))
    I_init = np.asarray(currents0)
    I_final = np.asarray(currents1)
    w = 0.38
    ax.bar(idx - w / 2, I_init, width=w, label="init")
    ax.bar(idx + w / 2, I_final, width=w, label="final")
    ax.set_xlabel("Electrode index")
    ax.set_ylabel("Injected current [A] (projected to net zero)")
    ax.set_title("Electrode currents")
    ax.legend()
    savefig(fig, outdir / "optim" / "currents_bar.png", dpi=args.dpi)


if __name__ == "__main__":
    main()
