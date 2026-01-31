from __future__ import annotations

import jax.numpy as jnp

from .spectral import spectral_derivative
from .torus import TorusSurface


def surface_current_from_current_potential(surface: TorusSurface, Phi: jnp.ndarray) -> jnp.ndarray:
    """Surface current K (A/m) from a *single-valued* current potential Phi (A).

    This is the REGCOIL / "current potential" model:
      K = n_hat × ∇_s Phi

    On a circular torus with coordinates (θ, φ) and F=0, this can be written as:
      K = (Phi_θ * r_φ - Phi_φ * r_θ) / sqrt(g)

    where r_θ and r_φ are the surface tangent vectors and sqrt(g)=norm(r_θ×r_φ).
    """
    Phi_theta = spectral_derivative(Phi, surface.k_theta, axis=0)
    Phi_phi = spectral_derivative(Phi, surface.k_phi, axis=1)
    denom = surface.sqrt_g[..., None] + 1e-30
    return (Phi_theta[..., None] * surface.r_phi - Phi_phi[..., None] * surface.r_theta) / denom


def surface_current_from_current_potential_with_net_currents(
    surface: TorusSurface,
    Phi_single_valued: jnp.ndarray,
    *,
    net_poloidal_current_A: float = 0.0,
    net_toroidal_current_A: float = 0.0,
) -> jnp.ndarray:
    """Surface current from current potential with net poloidal/toroidal currents.

    The net-current contributions correspond to *multi-valued* secular terms in Phi,
    which must NOT be differentiated using FFTs. Instead we add their derivatives
    analytically: Phi_theta adds Itor/(2π) and Phi_phi adds Ipol/(2π).
    """
    twopi = 2.0 * jnp.pi
    Phi_theta = spectral_derivative(Phi_single_valued, surface.k_theta, axis=0) + (
        net_toroidal_current_A / twopi
    )
    Phi_phi = spectral_derivative(Phi_single_valued, surface.k_phi, axis=1) + (
        net_poloidal_current_A / twopi
    )
    denom = surface.sqrt_g[..., None] + 1e-30
    return (Phi_theta[..., None] * surface.r_phi - Phi_phi[..., None] * surface.r_theta) / denom


def add_secular_current_terms_for_visualization(
    surface: TorusSurface, *, net_poloidal_current_A: float = 0.0, net_toroidal_current_A: float = 0.0
) -> jnp.ndarray:
    """Return a *multi-valued* Phi_sec(θ,φ) for visualization.

    The current potential can be multi-valued; its derivatives define K.

    Convention (matches REGCOIL naming):
      - net_poloidal_current_A multiplies φ / (2π) and yields poloidal surface current K_θ.
      - net_toroidal_current_A multiplies θ / (2π) and yields toroidal surface current K_φ.

    Warning: Do not pass this function's output to `surface_current_from_current_potential`,
    since the secular terms are not periodic/smooth and FFT derivatives will be wrong.
    """
    twopi = 2.0 * jnp.pi
    return (net_toroidal_current_A / twopi) * surface.theta[:, None] + (
        net_poloidal_current_A / twopi
    ) * surface.phi[None, :]


# Backwards-compatible alias (visualization-only).
add_secular_current_terms = add_secular_current_terms_for_visualization
