#!/usr/bin/env python3
"""Interactive 3D GUI: add sources/sinks on a torus and view currents + field lines.

Run (from `torus_solver/`):
  python examples/gui_torus_interactive.py
"""

from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--R0", type=float, default=3.0, help="Major radius [m]")
    p.add_argument("--a", type=float, default=1.0, help="Minor radius [m]")
    p.add_argument("--n-theta", type=int, default=32)
    p.add_argument("--n-phi", type=int, default=32)
    p.add_argument("--n-electrodes-max", type=int, default=32)
    p.add_argument("--I0", type=float, default=1000.0, help="Default electrode current [A]")
    p.add_argument("--Imax", type=float, default=3000.0, help="Slider range [A]")
    p.add_argument("--n-fieldlines", type=int, default=12)
    p.add_argument("--fieldline-steps", type=int, default=500)
    p.add_argument("--ds", type=float, default=0.03, help="Fieldline step size [m]")
    p.add_argument("--surface-opacity", type=float, default=0.35)
    args = p.parse_args()

    try:
        from torus_solver.gui_vtk import GUIConfig, run_torus_electrode_gui
    except ImportError as e:
        raise SystemExit(str(e)) from e

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
        surface_opacity=args.surface_opacity,
    )
    run_torus_electrode_gui(cfg=cfg)


if __name__ == "__main__":
    main()

