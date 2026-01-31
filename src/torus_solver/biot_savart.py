from __future__ import annotations

import jax
import jax.numpy as jnp

from .torus import TorusSurface


MU0 = 4e-7 * jnp.pi


def biot_savart_surface(
    surface: TorusSurface,
    K: jnp.ndarray,
    eval_points: jnp.ndarray,
    *,
    mu0: float = MU0,
    eps: float = 1e-9,
    chunk_size: int | None = 256,
) -> jnp.ndarray:
    """Magnetic field B (Tesla) from a surface current density K (A/m).

    In continuous form:

      B(x) = μ0/(4π) ∬ K(r') × (x-r') / norm(x-r')^3 dA'

    Notes:
      - `eps` is a small softening length to avoid numerical issues when points are
        extremely close to the surface.
      - This routine is JAX-differentiable w.r.t. `K` and `eval_points`.
    """
    eval_points = jnp.asarray(eval_points)
    r_surf = surface.r.reshape((-1, 3))
    K_surf = jnp.asarray(K).reshape((-1, 3))
    dA = surface.area_weights.reshape((-1,))

    pref = mu0 / (4 * jnp.pi)
    eps2 = eps * eps

    def field_chunk(P: jnp.ndarray) -> jnp.ndarray:
        # P: (N,3)
        R = P[:, None, :] - r_surf[None, :, :]  # (N,M,3)
        r2 = jnp.sum(R * R, axis=-1) + eps2  # (N,M)
        inv_r3 = r2 ** (-1.5)
        dB = jnp.cross(K_surf[None, :, :], R) * inv_r3[..., None] * dA[None, :, None]
        return pref * jnp.sum(dB, axis=1)  # (N,3)

    # For small point counts, the vmap path is fine; for larger problems, chunked
    # evaluation avoids allocating an (N_eval, N_surface, 3) tensor that can become
    # large in double precision.
    n_eval = eval_points.shape[0]
    if chunk_size is None or n_eval <= chunk_size:
        return field_chunk(eval_points)

    n_chunks = (n_eval + chunk_size - 1) // chunk_size
    pad = n_chunks * chunk_size - n_eval
    eval_pad = jnp.pad(eval_points, ((0, pad), (0, 0)))
    eval_chunks = eval_pad.reshape((n_chunks, chunk_size, 3))

    def scan_step(_, pts):
        return None, field_chunk(pts)

    _, B_chunks = jax.lax.scan(scan_step, None, eval_chunks)
    B_pad = B_chunks.reshape((n_chunks * chunk_size, 3))
    return B_pad[:n_eval, :]
