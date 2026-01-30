from __future__ import annotations

import jax.numpy as jnp

from .poisson import area_mean
from .torus import TorusSurface


def wrap_angle(x: jnp.ndarray) -> jnp.ndarray:
    """Wrap angles to (-π, π] in a JAX-differentiable way."""
    twopi = 2.0 * jnp.pi
    return jnp.mod(x + jnp.pi, twopi) - jnp.pi


def deposit_current_sources(
    surface: TorusSurface,
    *,
    theta_src: jnp.ndarray,  # (Ns,)
    phi_src: jnp.ndarray,  # (Ns,)
    currents: jnp.ndarray,  # (Ns,)
    sigma_theta: float,
    sigma_phi: float,
) -> jnp.ndarray:
    """Deposit smooth current source/sink density s(θ,φ) (A/m^2) onto the surface grid.

    Each source i injects a total current `currents[i]` distributed by a periodic
    Gaussian kernel in (θ, φ), normalized so that ∫ s dA = Σ currents.

    For magnetostatics, net injected current must be zero. This routine enforces
    a zero area-mean source density numerically.
    """
    theta_src = jnp.asarray(theta_src)
    phi_src = jnp.asarray(phi_src)
    currents = jnp.asarray(currents)

    dtheta = wrap_angle(surface.theta[:, None] - theta_src[None, :])  # (Nθ,Ns)
    dphi = wrap_angle(surface.phi[:, None] - phi_src[None, :])  # (Nφ,Ns)

    k_theta = jnp.exp(-0.5 * (dtheta / sigma_theta) ** 2)  # (Nθ,Ns)
    k_phi = jnp.exp(-0.5 * (dphi / sigma_phi) ** 2)  # (Nφ,Ns)

    # Normalize each separable kernel so that its integral over the surface is 1:
    #   norm_i = (Σθ kθ(θ,i) √g(θ) dθ) * (Σφ kφ(φ,i) dφ)
    # Since √g depends only on θ for a circular torus, we avoid materializing
    # the full (Nθ,Nφ,Ns) kernel.
    w_theta = (surface.sqrt_g[:, 0] * surface.dtheta)[:, None]  # (Nθ,1)
    norm_theta = jnp.sum(k_theta * w_theta, axis=0)  # (Ns,)
    norm_phi = jnp.sum(k_phi, axis=0) * surface.dphi  # (Ns,)
    norm = norm_theta * norm_phi  # (Ns,)

    # s(θ,φ) = Σ_i currents_i * kθ(θ,i) * kφ(φ,i) / norm_i
    a_theta = k_theta * (currents / norm)[None, :]  # (Nθ,Ns)
    s = a_theta @ k_phi.T  # (Nθ,Nφ)
    return s - area_mean(surface, s)
