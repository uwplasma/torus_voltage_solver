from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg

from .spectral import spectral_derivative, spectral_second_derivative
from .torus import TorusSurface


def area_mean(surface: TorusSurface, f: jnp.ndarray) -> jnp.ndarray:
    """Area-weighted mean of a scalar field on the surface."""
    return jnp.sum(f * surface.area_weights) / jnp.sum(surface.area_weights)


def laplace_beltrami_torus(surface: TorusSurface, V: jnp.ndarray) -> jnp.ndarray:
    """Laplace–Beltrami operator on a circular torus using spectral derivatives."""
    dV_dtheta = spectral_derivative(V, surface.k_theta, axis=0)
    term = surface.R * dV_dtheta
    dterm_dtheta = spectral_derivative(term, surface.k_theta, axis=0)
    term1 = dterm_dtheta / (surface.a * surface.a * surface.R)

    d2V_dphi2 = spectral_second_derivative(V, surface.k_phi, axis=1)
    term2 = d2V_dphi2 / surface.G
    return term1 + term2


def solve_current_potential(
    surface: TorusSurface,
    source_density: jnp.ndarray,
    *,
    tol: float = 1e-10,
    maxiter: int = 2_000,
    use_preconditioner: bool = False,
) -> Tuple[jnp.ndarray, int]:
    """Solve -∇²V = source_density for V on the torus (gauge fixed to ⟨V⟩=0).

    Note: `use_preconditioner=True` enables a simple spectral preconditioner based on a
    constant-coefficient approximation. It is experimental and may or may not improve
    convergence depending on the right-hand side and resolution.
    """
    b = source_density - area_mean(surface, source_density)

    def matvec(x):
        return -laplace_beltrami_torus(surface, x) + area_mean(surface, x)

    def preconditioner(x):
        # Spectral inverse of a constant-coefficient approximation:
        #   A0 = -(∂θθ/a^2 + ∂φφ/R0^2) + mean
        # which is diagonal in the (θ,φ) Fourier basis.
        k_theta2 = (surface.k_theta**2)[:, None] / (surface.a * surface.a)
        k_phi2 = (surface.k_phi**2)[None, :] / (surface.R0 * surface.R0)
        denom = k_theta2 + k_phi2
        denom = denom.at[0, 0].set(1.0)
        X = jnp.fft.fftn(x, axes=(0, 1))
        Y = X / denom
        y = jnp.fft.ifftn(Y, axes=(0, 1)).real
        return y

    x0 = jnp.zeros_like(b)
    M = preconditioner if use_preconditioner else None
    V, info = cg(matvec, b, x0=x0, tol=tol, maxiter=maxiter, M=M)
    V = V - area_mean(surface, V)
    return V, info


def surface_current_from_potential(
    surface: TorusSurface, V: jnp.ndarray, *, sigma_s: float = 1.0
) -> jnp.ndarray:
    """Surface current density K (A/m) from electric potential V: K = -σ_s ∇_s V."""
    dV_dtheta = spectral_derivative(V, surface.k_theta, axis=0)
    dV_dphi = spectral_derivative(V, surface.k_phi, axis=1)

    # Surface gradient ∇_s V = (1/a^2) V_θ r_θ + (1/G) V_φ r_φ
    grad = (dV_dtheta / (surface.a * surface.a))[..., None] * surface.r_theta + (
        dV_dphi / surface.G
    )[..., None] * surface.r_phi
    return (-sigma_s) * grad
