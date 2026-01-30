from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def _normalize(v: jnp.ndarray, *, eps: float = 1e-12) -> jnp.ndarray:
    return v / (jnp.linalg.norm(v, axis=-1, keepdims=True) + eps)


def trace_field_line(
    B_fn: Callable[[jnp.ndarray], jnp.ndarray],
    r0: jnp.ndarray,
    *,
    step_size: float,
    n_steps: int,
    normalize: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Trace a single magnetic field line using fixed-step RK4.

    The integration variable is an arbitrary "time" parameter. If `normalize=True`,
    we trace the unit vector b = B/|B| so the step size is roughly a spatial arc
    length (in meters if coordinates are meters).

    `B_fn` must accept points of shape (...,3) and return B of shape (...,3).
    """
    r0 = jnp.asarray(r0, dtype=jnp.float64)

    def f(r: jnp.ndarray) -> jnp.ndarray:
        B = B_fn(r[None, :])[0]
        if normalize:
            return _normalize(B, eps=eps)
        return B

    h = jnp.array(step_size, dtype=jnp.float64)

    def rk4_step(r, _):
        k1 = f(r)
        k2 = f(r + 0.5 * h * k1)
        k3 = f(r + 0.5 * h * k2)
        k4 = f(r + h * k3)
        r_next = r + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return r_next, r_next

    rN, traj = jax.lax.scan(rk4_step, r0, xs=None, length=n_steps)
    del rN
    return jnp.concatenate([r0[None, :], traj], axis=0)


def trace_field_lines(
    B_fn: Callable[[jnp.ndarray], jnp.ndarray],
    r0: jnp.ndarray,
    *,
    step_size: float,
    n_steps: int,
    normalize: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Vectorized field-line tracing for a batch of initial points (N,3)."""
    r0 = jnp.asarray(r0, dtype=jnp.float64)
    return jax.vmap(
        lambda x: trace_field_line(
            B_fn, x, step_size=step_size, n_steps=n_steps, normalize=normalize, eps=eps
        )
    )(r0)


def trace_field_lines_batch(
    B_fn: Callable[[jnp.ndarray], jnp.ndarray],
    r0: jnp.ndarray,
    *,
    step_size: float,
    n_steps: int,
    normalize: bool = True,
    eps: float = 1e-12,
) -> jnp.ndarray:
    """Trace multiple field lines in one `lax.scan` (faster than vmap-of-scan).

    `r0` has shape (N,3). `B_fn` must accept (N,3) and return (N,3).

    Returns an array of shape (n_steps+1, N, 3).
    """
    r0 = jnp.asarray(r0, dtype=jnp.float64)

    def f(r: jnp.ndarray) -> jnp.ndarray:
        B = B_fn(r)
        if normalize:
            return _normalize(B, eps=eps)
        return B

    h = jnp.array(step_size, dtype=jnp.float64)

    def rk4_step(r, _):
        k1 = f(r)
        k2 = f(r + 0.5 * h * k1)
        k3 = f(r + 0.5 * h * k2)
        k4 = f(r + h * k3)
        r_next = r + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return r_next, r_next

    _, traj = jax.lax.scan(rk4_step, r0, xs=None, length=n_steps)
    return jnp.concatenate([r0[None, ...], traj], axis=0)
