#!/usr/bin/env python3
"""Benchmark: field-line tracing performance in analytic tokamak-like fields.

Run:
  - `python benchmarks/bench_fieldline.py`
"""

from __future__ import annotations

if __package__ in (None, ""):
    import pathlib
    import sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import time

import jax
import jax.numpy as jnp

from torus_solver.fieldline import trace_field_lines
from torus_solver.fields import tokamak_like_field


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    R0 = 3.0
    Btor0 = 1.0
    Bpol0 = 0.1

    n_lines = 16
    n_steps = 5000
    step = 0.01

    phi0 = jnp.linspace(0.0, 2 * jnp.pi, n_lines, endpoint=False)
    r0 = jnp.stack([(R0 + 0.4) * jnp.cos(phi0), (R0 + 0.4) * jnp.sin(phi0), 0.0 * phi0], axis=-1)

    def B_fn(xyz):
        return tokamak_like_field(xyz, B_tor0=Btor0, B_pol0=Bpol0, R0=R0)

    traced = jax.jit(lambda x: trace_field_lines(B_fn, x, step_size=step, n_steps=n_steps))

    t0 = time.perf_counter()
    pts = traced(r0)
    pts.block_until_ready()
    t1 = time.perf_counter()
    print(f"compile+run_s: {t1 - t0:.3f}  shape={pts.shape}")

    n_iter = 3
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pts = traced(r0)
    pts.block_until_ready()
    t1 = time.perf_counter()
    print(f"avg_run_s: {(t1 - t0) / n_iter:.3f} over {n_iter} iters")


if __name__ == "__main__":
    main()
