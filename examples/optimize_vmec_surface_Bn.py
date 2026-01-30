#!/usr/bin/env python3
"""Example: optimize electrode sources/sinks on a circular torus to reduce B·n on a VMEC surface.

Goal (REGCOIL-like objective, but with discrete electrodes):
  - Given a target "plasma" boundary surface from a VMEC input file, minimize the
    *normal* component of the total magnetic field on that surface:

        L = ⟨ (B_total · n_hat)^2 ⟩_surface  +  λ ⟨I^2⟩

  - Here B_total = B_bg + B_shell, where:
      * B_bg is an imposed background toroidal field ~ 1/R (analytic).
      * B_shell is the Biot–Savart field from surface currents on the torus induced
        by electrode current sources/sinks (our model).

Important notes:
  - This is a *prototype* objective. Driving B·n≈0 is a necessary (but not sufficient)
    ingredient for a magnetic surface.
  - The VMEC boundary is shifted/scaled (if needed) so it fits inside the circular torus.

Run (from `torus_solver/`):
  python examples/optimize_vmec_surface_Bn.py --vmec-input examples/input.QA_nfp2
"""

from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from torus_solver import make_torus_surface
from torus_solver.biot_savart import biot_savart_surface
from torus_solver.fieldline import trace_field_lines_batch
from torus_solver.fields import ideal_toroidal_field
from torus_solver.optimize import SourceParams, enforce_net_zero, surface_solution
from torus_solver.plotting import (
    ensure_dir,
    plot_3d_torus,
    plot_fieldlines_3d,
    plot_loss_history,
    plot_surface_map,
    savefig,
    set_plot_style,
)
from torus_solver.vmec import read_vmec_boundary, vmec_boundary_RZ_and_derivatives
import torus_solver.plotting as tplot


def _wrap_angle_np(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2 * np.pi) - np.pi


def _resolve_vmec_input(vmec_input: str) -> Path:
    p = Path(vmec_input)
    if p.exists():
        return p

    # If the user passed e.g. "input.QA_nfp2" while running from `torus_solver/`,
    # the file lives next to this script in `examples/`.
    script_dir = Path(__file__).resolve().parent
    cand = script_dir / vmec_input
    if cand.exists():
        return cand

    msg = (
        "VMEC input file not found.\n"
        f"  got: {vmec_input}\n"
        f"  tried: {p.resolve()}\n"
        f"  tried: {cand}\n"
        "Tip: run with `--vmec-input examples/input.QA_nfp2` (from `torus_solver/`)."
    )
    raise FileNotFoundError(msg)


def _fit_surface_into_torus(
    *,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    R_theta: jnp.ndarray,
    R_phi: jnp.ndarray,
    Z_theta: jnp.ndarray,
    Z_phi: jnp.ndarray,
    torus_R0: float,
    torus_a: float,
    fit_margin: float,
) -> tuple[
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    jnp.ndarray,
    float,
    float,
    float,
]:
    """Shift and (if needed) scale a VMEC surface so it fits inside the torus."""
    # Shift so mean(R) matches the torus major radius.
    R_mean = jnp.mean(R)
    shift = float(torus_R0 - float(R_mean))
    R_shift = R + shift

    dR = R_shift - torus_R0
    rho = jnp.sqrt(dR * dR + Z * Z + 1e-30)
    rho_max = float(jnp.max(rho))

    target = float(fit_margin * torus_a)
    scale = 1.0
    if rho_max > target:
        scale = target / rho_max

    R_fit = torus_R0 + scale * dR
    Z_fit = scale * Z
    R_theta_fit = scale * R_theta
    R_phi_fit = scale * R_phi
    Z_theta_fit = scale * Z_theta
    Z_phi_fit = scale * Z_phi
    return (
        R_fit,
        Z_fit,
        R_theta_fit,
        R_phi_fit,
        Z_theta_fit,
        Z_phi_fit,
        shift,
        scale,
        rho_max,
    )


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    p = argparse.ArgumentParser()
    p.add_argument("--vmec-input", type=str, default="input.QA_nfp2")

    # Winding surface (circular torus) geometry.
    p.add_argument("--R0", type=float, default=1.0, help="Torus major radius [m]")
    p.add_argument("--a", type=float, default=0.5, help="Torus minor radius [m]")
    p.add_argument("--n-theta", type=int, default=48, help="Torus grid Nθ")
    p.add_argument("--n-phi", type=int, default=48, help="Torus grid Nφ")

    # Target surface sampling.
    p.add_argument("--surf-n-theta", type=int, default=16, help="Target surface sampling Nθ")
    p.add_argument("--surf-n-phi", type=int, default=28, help="Target surface sampling Nφ")
    p.add_argument("--fit-margin", type=float, default=0.92, help="Require rho_max <= fit_margin * a")

    # Background field (analytic toroidal 1/R).
    p.add_argument("--B0", type=float, default=1, help="Background toroidal field at R=R0 [T]")

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
    p.add_argument("--n-steps", type=int, default=50)
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

    # Fieldline plot controls.
    p.add_argument("--no-fieldlines", action="store_true")
    p.add_argument("--n-fieldlines", type=int, default=12)
    p.add_argument("--fieldline-steps", type=int, default=600)
    p.add_argument("--fieldline-ds", type=float, default=0.02)
    p.add_argument("--biot-savart-eps", type=float, default=1e-8)

    # Outputs.
    p.add_argument("--outdir", type=str, default="figures/optimize_vmec_surface_Bn")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--plot-stride", type=int, default=3)
    args = p.parse_args()

    if not args.no_plots:
        set_plot_style()
        outdir = ensure_dir(args.outdir)
        ensure_dir(outdir / "target")
        ensure_dir(outdir / "surface")
        ensure_dir(outdir / "optim")
        ensure_dir(outdir / "geometry")
        ensure_dir(outdir / "fieldlines")
        print(f"Saving figures to: {outdir}")

    vmec_path = _resolve_vmec_input(args.vmec_input)
    print("Loading VMEC boundary:", str(vmec_path))
    boundary = read_vmec_boundary(str(vmec_path))
    print(f"  parsed NFP={boundary.nfp}  nmodes={boundary.m.size}")

    # Target surface grid (full torus: phi in [0,2π)).
    theta_s = jnp.linspace(0.0, 2 * jnp.pi, int(args.surf_n_theta), endpoint=False, dtype=jnp.float64)
    phi_s = jnp.linspace(0.0, 2 * jnp.pi, int(args.surf_n_phi), endpoint=False, dtype=jnp.float64)
    dtheta = float(2 * np.pi / int(args.surf_n_theta))
    dphi = float(2 * np.pi / int(args.surf_n_phi))

    # Evaluate VMEC surface + derivatives.
    R, Z, R_th, R_ph, Z_th, Z_ph = vmec_boundary_RZ_and_derivatives(boundary, theta=theta_s, phi=phi_s)
    (
        R_fit,
        Z_fit,
        R_th_fit,
        R_ph_fit,
        Z_th_fit,
        Z_ph_fit,
        shift_R,
        scale_rho,
        rho_max,
    ) = _fit_surface_into_torus(
        R=R,
        Z=Z,
        R_theta=R_th,
        R_phi=R_ph,
        Z_theta=Z_th,
        Z_phi=Z_ph,
        torus_R0=float(args.R0),
        torus_a=float(args.a),
        fit_margin=float(args.fit_margin),
    )

    print("Fitting target surface into the circular torus")
    print(f"  torus: R0={args.R0} a={args.a}  fit_margin={args.fit_margin}")
    print(f"  shift_R = {shift_R:+.6e} m")
    print(f"  rho_max(before) = {rho_max:.6e} m   => scale={scale_rho:.6e}")
    print(f"  rho_max(after)  = {rho_max*scale_rho:.6e} m  (target <= {args.fit_margin*args.a:.6e} m)")

    # Build xyz + normals on the fitted target surface.
    phi2 = phi_s[None, :]
    c = jnp.cos(phi2)
    s = jnp.sin(phi2)

    x = R_fit * c
    y = R_fit * s
    z = Z_fit
    xyz = jnp.stack([x, y, z], axis=-1)  # (Nθ,Nφ,3)

    r_theta = jnp.stack([R_th_fit * c, R_th_fit * s, Z_th_fit], axis=-1)
    r_phi = jnp.stack([R_ph_fit * c - R_fit * s, R_ph_fit * s + R_fit * c, Z_ph_fit], axis=-1)
    n_vec = jnp.cross(r_theta, r_phi)
    n_norm = jnp.linalg.norm(n_vec, axis=-1)
    n_hat = n_vec / (n_norm[..., None] + 1e-30)
    w_area = n_norm  # dθ dφ cancels in normalized weighted mean

    eval_points = xyz.reshape((-1, 3))
    normals = n_hat.reshape((-1, 3))
    weights = w_area.reshape((-1,))

    print("Building winding surface (circular torus)")
    surface = make_torus_surface(R0=args.R0, a=args.a, n_theta=args.n_theta, n_phi=args.n_phi)

    # Precompute background field at the evaluation points.
    B_bg = ideal_toroidal_field(eval_points, B0=args.B0, R0=args.R0)
    Bn_bg = jnp.sum(B_bg * normals, axis=-1)
    Bn_bg_rms = float(jnp.sqrt(jnp.sum(weights * (Bn_bg * Bn_bg)) / jnp.sum(weights)))

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
    print(f"  objective:  minimize area-weighted <(B·n)^2> on target surface")
    print(f"  background: ideal toroidal B0={args.B0} T at R0={args.R0}")
    print(f"  current_scale: {args.current_scale:.6e} A per unit(currents_raw)")
    print(f"  optim:      steps={args.n_steps} lr={args.lr} regI={args.reg_currents} regPos={args.reg_positions}")
    print(f"  cg:         tol={args.cg_tol} maxiter={args.cg_maxiter} preconditioner={bool(args.use_preconditioner)}")

    B_scale = float(abs(args.B0)) if args.B0 != 0 else 1.0
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
        Bn = jnp.sum(B_tot * normals, axis=-1)

        # Area-weighted mean square normal field.
        loss_bn = jnp.sum(weights * (Bn / B_scale) ** 2) / jnp.sum(weights)

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
            "Bn_rms_T": jnp.sqrt(jnp.sum(weights * (Bn * Bn)) / jnp.sum(weights)),
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
    history = {"loss": [], "loss_bn": [], "loss_reg": [], "Bn_rms_T": [], "I_rms_A": []}

    print("Initial scaling diagnostics on target surface (area-weighted)")
    print(f"  Bn_bg_rms      = {Bn_bg_rms:.3e} T")
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
    Bn_shell_init = jnp.sum(B_shell_init * normals, axis=-1)
    Bn_shell_rms = float(jnp.sqrt(jnp.sum(weights * (Bn_shell_init * Bn_shell_init)) / jnp.sum(weights)))
    print(f"  Bn_shell_rms   = {Bn_shell_rms:.3e} T")
    print(f"  ratio(shell/bg)= {Bn_shell_rms / (Bn_bg_rms + 1e-30):.3e}")

    t0 = time.perf_counter()
    for k in range(int(args.n_steps)):
        params, state, aux = step(params, state)
        if k % 25 == 0 or k == int(args.n_steps) - 1:
            print(
                "step={:4d} loss={:.6e} bn={:.3e} reg={:.3e}  Bn_rms={:.3e}T  I_rms={:.3e}A".format(
                    k,
                    float(aux["loss"]),
                    float(aux["loss_bn"]),
                    float(aux["loss_reg"]),
                    float(aux["Bn_rms_T"]),
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
        return ideal_toroidal_field(pts, B0=args.B0, R0=args.R0) + biot_savart_surface(
            surface, K, pts, eps=float(args.biot_savart_eps)
        )

    B0_tot = B_total_from_K(K0, eval_points)
    B1_tot = B_total_from_K(K1, eval_points)
    Bn0 = jnp.sum(B0_tot * normals, axis=-1)
    Bn1 = jnp.sum(B1_tot * normals, axis=-1)

    def wrms(x):
        return float(jnp.sqrt(jnp.sum(weights * (x * x)) / jnp.sum(weights)))

    print("Bn summary on target surface (area-weighted)")
    print(f"  Bn_rms(init)  = {wrms(Bn0):.3e} T")
    print(f"  Bn_rms(final) = {wrms(Bn1):.3e} T")

    if args.no_plots:
        return

    outdir = ensure_dir(args.outdir)

    # Target surface maps.
    Bn0_grid = np.asarray(Bn0.reshape((int(args.surf_n_theta), int(args.surf_n_phi))))
    Bn1_grid = np.asarray(Bn1.reshape((int(args.surf_n_theta), int(args.surf_n_phi))))
    plot_surface_map(
        phi=np.asarray(phi_s),
        theta=np.asarray(theta_s),
        data=Bn0_grid,
        title=r"Target surface $B\cdot \hat n$ (init)",
        cbar_label=r"$B\cdot \hat n$ [T]",
        cmap="coolwarm",
        path=outdir / "target" / "Bn_init.png",
    )
    plot_surface_map(
        phi=np.asarray(phi_s),
        theta=np.asarray(theta_s),
        data=Bn1_grid,
        title=r"Target surface $B\cdot \hat n$ (final)",
        cbar_label=r"$B\cdot \hat n$ [T]",
        cmap="coolwarm",
        path=outdir / "target" / "Bn_final.png",
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
        metrics={"Bn_rms [T]": history["Bn_rms_T"], "I_rms [A]": history["I_rms_A"]},
        title="Bn_rms and I_rms",
        path=outdir / "optim" / "Bn_I_rms.png",
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
    if not args.no_fieldlines:
        print("Tracing field lines for final total field (background + Biot–Savart)...")
        theta_seed = jnp.linspace(0.0, 2 * jnp.pi, int(args.n_fieldlines), endpoint=False)
        rho = 0.5 * surface.a
        R_seed = surface.R0 + rho * jnp.cos(theta_seed)
        Z_seed = rho * jnp.sin(theta_seed)
        seeds = jnp.stack([R_seed, jnp.zeros_like(R_seed), Z_seed], axis=-1)

        def B_fn(xyz_pts: jnp.ndarray) -> jnp.ndarray:
            return ideal_toroidal_field(xyz_pts, B0=args.B0, R0=args.R0) + biot_savart_surface(
                surface, K1, xyz_pts, eps=float(args.biot_savart_eps)
            )

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

        plot_fieldlines_3d(
            torus_xyz=np.asarray(surface.r),
            traj=np.asarray(traj),
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
