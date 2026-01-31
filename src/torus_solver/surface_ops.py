from __future__ import annotations

import jax.numpy as jnp

from .spectral import spectral_derivative
from .torus import TorusSurface


def contravariant_components_torus(surface: TorusSurface, v_xyz: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return contravariant components (v^θ, v^φ) of a tangential vector field on the torus.

    The torus parameterization has F=0 and metric coefficients E=a^2 and G=(R0+a cosθ)^2,
    so v · r_θ = E v^θ and v · r_φ = G v^φ.
    """
    v_xyz = jnp.asarray(v_xyz)
    v_dot_rtheta = jnp.sum(v_xyz * surface.r_theta, axis=-1)
    v_dot_rphi = jnp.sum(v_xyz * surface.r_phi, axis=-1)
    v_theta = v_dot_rtheta / (surface.a * surface.a)
    v_phi = v_dot_rphi / surface.G
    return v_theta, v_phi


def surface_divergence_torus(surface: TorusSurface, v_xyz: jnp.ndarray) -> jnp.ndarray:
    """Surface divergence of a tangential vector field on the circular torus.

    Using contravariant components (v^θ, v^φ):
      div_s v = (1/sqrt(g)) [ ∂_θ (sqrt(g) v^θ) + ∂_φ (sqrt(g) v^φ) ].
    """
    v_theta, v_phi = contravariant_components_torus(surface, v_xyz)
    t1 = spectral_derivative(surface.sqrt_g * v_theta, surface.k_theta, axis=0)
    t2 = spectral_derivative(surface.sqrt_g * v_phi, surface.k_phi, axis=1)
    return (t1 + t2) / (surface.sqrt_g + 1e-30)
