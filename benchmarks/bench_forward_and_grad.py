#!/usr/bin/env python3
"""Micro-benchmark for the torus source/sink forward model (and its gradient).

Run:
  - `python benchmarks/bench_forward_and_grad.py`
"""

from __future__ import annotations

if __package__ in (None, ""):
    import pathlib, sys

    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import time

import jax
import jax.numpy as jnp

from torus_solver import make_torus_surface
from torus_solver.optimize import SourceParams, forward_B, make_helical_axis_points


def main() -> None:
    jax.config.update("jax_enable_x64", True)

    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=64, n_phi=64)
    _, points, _, e_phi, _ = make_helical_axis_points(R_axis=surface.R0, n_points=64)
    B_target = 1.0 * e_phi

    n_sources = 16
    key = jax.random.key(0)
    params = SourceParams(
        theta_src=2 * jnp.pi * jax.random.uniform(key, (n_sources,)),
        phi_src=2 * jnp.pi * jax.random.uniform(jax.random.fold_in(key, 1), (n_sources,)),
        currents_raw=0.1 * jax.random.normal(jax.random.fold_in(key, 2), (n_sources,)),
    )

    sigma_theta = 0.22
    sigma_phi = 0.22

    def loss_fn(p: SourceParams) -> jnp.ndarray:
        B = forward_B(
            surface, p, eval_points=points, sigma_theta=sigma_theta, sigma_phi=sigma_phi
        )
        err = B - B_target
        return jnp.mean(jnp.sum(err * err, axis=-1))

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # Compile + first run
    t0 = time.perf_counter()
    loss0, g0 = loss_and_grad(params)
    loss0.block_until_ready()
    _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), g0)
    t1 = time.perf_counter()
    print(f"compile+run_s: {t1 - t0:.3f}  loss0: {float(loss0):.6e}")

    # Steady-state runs
    n_iter = 5
    t0 = time.perf_counter()
    for _ in range(n_iter):
        loss, g = loss_and_grad(params)
    loss.block_until_ready()
    _ = jax.tree_util.tree_map(lambda x: x.block_until_ready(), g)
    t1 = time.perf_counter()
    print(f"avg_run_s: {(t1 - t0) / n_iter:.3f} over {n_iter} iters")


if __name__ == "__main__":
    main()
