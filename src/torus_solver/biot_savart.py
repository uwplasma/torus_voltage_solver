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
) -> jnp.ndarray:
    """Magnetic field B (Tesla) from a surface current density K (A/m).

    B(x) = μ0/(4π) ∬ K(r') × (x-r') / |x-r'|^3 dA'

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

    def field_at(p):
        R = p[None, :] - r_surf  # (M,3)
        r2 = jnp.sum(R * R, axis=-1) + eps2
        inv_r3 = r2 ** (-1.5)
        dB = jnp.cross(K_surf, R) * inv_r3[:, None] * dA[:, None]
        return pref * jnp.sum(dB, axis=0)

    return jax.vmap(field_at)(eval_points)
