#!/usr/bin/env python3
"""Interactive 3D GUI demo: toroidal cut voltage + optional electrode sources/sinks.

This demo uses an axisymmetric *cut voltage* model: a voltage drop across a toroidal
cut makes the potential multi-valued and drives a poloidal surface current, producing
a toroidal field inside the torus.

You can also add *extra* sources/sinks (electrodes) on top of the cut-driven current
using the same hotkeys as the electrode GUI.

Run (from the repo root, `torus_voltage_solver/`):
  python examples/2_intermediate/gui_torus_demo_inboard_cut.py
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

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0, help="Major radius [m]")
    p.add_argument("--a", type=float, default=0.3, help="Minor radius [m]")
    p.add_argument("--n-theta", type=int, default=32)
    p.add_argument("--n-phi", type=int, default=32)
    p.add_argument("--theta-cut", type=float, default=float(np.pi))
    p.add_argument("--sigma-s", type=float, default=1.0, help="Surface conductivity scale")
    p.add_argument("--Vcut", type=float, default=1.0, help="Initial cut voltage [arb]")
    p.add_argument("--Vmax", type=float, default=5.0, help="Slider range for Vcut [arb]")
    p.add_argument("--n-electrodes-max", type=int, default=32)
    p.add_argument("--I0", type=float, default=1000.0, help="Default added electrode current [A]")
    p.add_argument("--Imax", type=float, default=3000.0, help="Slider range for electrode current [A]")
    p.add_argument("--sigma-theta", type=float, default=0.25)
    p.add_argument("--sigma-phi", type=float, default=0.25)
    p.add_argument("--cg-tol", type=float, default=1e-8)
    p.add_argument("--cg-maxiter", type=int, default=800)
    p.add_argument("--n-fieldlines", type=int, default=12)
    p.add_argument("--fieldline-steps", type=int, default=500)
    p.add_argument("--ds", type=float, default=0.03, help="Fieldline step size [m]")
    p.add_argument(
        "--Bext0",
        type=float,
        default=0.0,
        help="Optional background toroidal field at R=R0 [T] (ideal 1/R).",
    )
    p.add_argument(
        "--Bpol0",
        type=float,
        default=0.0,
        help="Optional background poloidal field at R=R0 [T] (tokamak-like 1/R).",
    )
    p.add_argument("--surface-opacity", type=float, default=0.35)
    args = p.parse_args()

    try:
        from torus_solver.gui_vtk import CutGUIConfig, run_torus_cut_voltage_gui
    except ImportError as e:
        raise SystemExit(str(e)) from e

    cfg = CutGUIConfig(
        R0=args.R0,
        a=args.a,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        theta_cut=args.theta_cut,
        sigma_s=args.sigma_s,
        V_cut_default=args.Vcut,
        V_cut_slider_max=args.Vmax,
        n_electrodes_max=args.n_electrodes_max,
        current_default_A=args.I0,
        current_slider_max_A=args.Imax,
        sigma_theta=args.sigma_theta,
        sigma_phi=args.sigma_phi,
        cg_tol=args.cg_tol,
        cg_maxiter=args.cg_maxiter,
        n_fieldlines=args.n_fieldlines,
        fieldline_steps=args.fieldline_steps,
        fieldline_step_size_m=args.ds,
        Bext0=args.Bext0,
        Bpol0=args.Bpol0,
        bg_field_default_on=(float(args.Bext0) != 0.0),
        bg_poloidal_default_on=(float(args.Bpol0) != 0.0),
        surface_opacity=args.surface_opacity,
    )

    run_torus_cut_voltage_gui(cfg=cfg)


if __name__ == "__main__":
    main()
