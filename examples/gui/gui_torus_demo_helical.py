#!/usr/bin/env python3
"""Interactive 3D GUI demo with a preloaded helical electrode pattern.

This is a *toy* starting point intended to show that the GUI can produce a
nontrivial 3D current pattern and field-line structure immediately.

Run (from the repo root, `torus_voltage_solver/`):
  python examples/gui/gui_torus_demo_helical.py
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
    p.add_argument("--a", type=float, default=1.0, help="Minor radius [m]")
    p.add_argument("--n-theta", type=int, default=32)
    p.add_argument("--n-phi", type=int, default=32)
    p.add_argument("--n-electrodes-max", type=int, default=32)
    p.add_argument("--n-demo", type=int, default=16, help="Number of preloaded electrodes")
    p.add_argument("--I0", type=float, default=1200.0, help="Current scale [A]")
    p.add_argument("--m", type=int, default=1, help="Helical pitch: theta ~ m*phi")
    p.add_argument("--nfp", type=int, default=2, help="Current modulation periodicity")
    p.add_argument("--Imax", type=float, default=3000.0, help="Slider range [A]")
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
        from torus_solver.gui_vtk import GUIConfig, run_torus_electrode_gui
    except ImportError as e:
        raise SystemExit(str(e)) from e

    n0 = min(args.n_demo, args.n_electrodes_max)
    phi = np.linspace(0.0, 2 * np.pi, n0, endpoint=False)
    theta = (args.m * phi) % (2 * np.pi)
    I = args.I0 * np.sin(args.nfp * phi)

    init = {"theta": theta, "phi": phi, "I": I}

    cfg = GUIConfig(
        R0=args.R0,
        a=args.a,
        n_theta=args.n_theta,
        n_phi=args.n_phi,
        n_electrodes_max=args.n_electrodes_max,
        current_default_A=args.I0,
        current_slider_max_A=args.Imax,
        n_fieldlines=args.n_fieldlines,
        fieldline_steps=args.fieldline_steps,
        fieldline_step_size_m=args.ds,
        Bext0=args.Bext0,
        Bpol0=args.Bpol0,
        bg_field_default_on=(float(args.Bext0) != 0.0),
        bg_poloidal_default_on=(float(args.Bpol0) != 0.0),
        surface_opacity=args.surface_opacity,
    )
    run_torus_electrode_gui(cfg=cfg, initial_electrodes=init)


if __name__ == "__main__":
    main()
