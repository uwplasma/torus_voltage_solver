#!/usr/bin/env python3
"""Interactive GUI: optimize electrode sources/sinks to reduce (B·n)/|B| on a VMEC surface.

Run (from `torus_solver/`):
  python examples/gui_optimize_vmec_surface_Bn.py --vmec-input examples/input.QA_nfp2
"""

from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vmec-input", type=str, default="input.QA_nfp2")

    p.add_argument("--R0", type=float, default=1.0)
    p.add_argument("--a", type=float, default=0.3)
    p.add_argument("--n-theta", type=int, default=32)
    p.add_argument("--n-phi", type=int, default=32)

    p.add_argument("--surf-n-theta", type=int, default=32)
    p.add_argument("--surf-n-phi", type=int, default=56)
    p.add_argument("--fit-margin", type=float, default=0.7)

    p.add_argument("--B0", type=float, default=1.0)
    p.add_argument("--sigma-theta", type=float, default=0.25)
    p.add_argument("--sigma-phi", type=float, default=0.25)
    p.add_argument("--sigma-s", type=float, default=1.0)

    p.add_argument("--n-electrodes-max", type=int, default=64)
    p.add_argument("--n-electrodes-init", type=int, default=32)
    p.add_argument("--init-current-raw-rms", type=float, default=1.0)
    p.add_argument("--current-scale", type=float, default=None)
    p.add_argument("--I0", type=float, default=1e6, help="Default current for added electrodes [A]")
    p.add_argument("--Imax", type=float, default=1e7, help="Slider range [A]")

    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--reg-currents", type=float, default=1e-3)
    p.add_argument("--steps-per-opt", type=int, default=25)
    p.add_argument("--no-optimize-positions", action="store_true")

    p.add_argument("--cg-tol", type=float, default=1e-10)
    p.add_argument("--cg-maxiter", type=int, default=2000)
    p.add_argument("--use-preconditioner", action="store_true")

    p.add_argument("--n-fieldlines", type=int, default=12)
    p.add_argument("--fieldline-steps", type=int, default=500)
    p.add_argument("--ds", type=float, default=0.03)
    p.add_argument("--biot-savart-eps", type=float, default=1e-8)

    p.add_argument("--surface-opacity", type=float, default=0.35)
    p.add_argument("--target-opacity", type=float, default=0.25)
    args = p.parse_args()

    try:
        from torus_solver.gui_vtk import VmecOptGUIConfig, run_torus_vmec_optimize_gui
    except ImportError as e:
        raise SystemExit(str(e)) from e

    cfg = VmecOptGUIConfig(
        vmec_input=args.vmec_input,
        surf_n_theta=args.surf_n_theta,
        surf_n_phi=args.surf_n_phi,
        fit_margin=args.fit_margin,
        R0=args.R0,
        a=args.a,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        B0=args.B0,
        n_electrodes_max=args.n_electrodes_max,
        n_electrodes_init=args.n_electrodes_init,
        current_default_A=args.I0,
        current_slider_max_A=args.Imax,
        sigma_theta=args.sigma_theta,
        sigma_phi=args.sigma_phi,
        sigma_s=args.sigma_s,
        current_scale=args.current_scale,
        init_current_raw_rms=args.init_current_raw_rms,
        lr=args.lr,
        reg_currents=args.reg_currents,
        optimize_positions=not bool(args.no_optimize_positions),
        steps_per_opt=args.steps_per_opt,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        use_preconditioner=bool(args.use_preconditioner),
        biot_savart_eps=args.biot_savart_eps,
        n_fieldlines=args.n_fieldlines,
        fieldline_steps=args.fieldline_steps,
        fieldline_step_size_m=args.ds,
        surface_opacity=args.surface_opacity,
        target_opacity=args.target_opacity,
    )
    run_torus_vmec_optimize_gui(cfg=cfg)


if __name__ == "__main__":
    main()
