from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .spectral import make_wavenumbers


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class TorusSurface:
    """A circular torus surface discretized on a (θ, φ) grid."""

    R0: float
    a: float

    theta: jnp.ndarray  # (Nθ,)
    phi: jnp.ndarray  # (Nφ,)
    dtheta: float
    dphi: float

    k_theta: jnp.ndarray  # (Nθ,)
    k_phi: jnp.ndarray  # (Nφ,)

    R: jnp.ndarray  # (Nθ,1) = R0 + a cosθ
    sqrt_g: jnp.ndarray  # (Nθ,1) = a (R0 + a cosθ)
    G: jnp.ndarray  # (Nθ,1) = (R0 + a cosθ)^2

    r: jnp.ndarray  # (Nθ,Nφ,3)
    r_theta: jnp.ndarray  # (Nθ,Nφ,3)
    r_phi: jnp.ndarray  # (Nθ,Nφ,3)

    area_weights: jnp.ndarray  # (Nθ,Nφ) = sqrt_g * dθ * dφ

    def tree_flatten(self):
        children = (
            self.theta,
            self.phi,
            self.k_theta,
            self.k_phi,
            self.R,
            self.sqrt_g,
            self.G,
            self.r,
            self.r_theta,
            self.r_phi,
            self.area_weights,
        )
        aux = (self.R0, self.a, self.dtheta, self.dphi)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (R0, a, dtheta, dphi) = aux
        (
            theta,
            phi,
            k_theta,
            k_phi,
            R,
            sqrt_g,
            G,
            r,
            r_theta,
            r_phi,
            area_weights,
        ) = children
        return cls(
            R0=R0,
            a=a,
            theta=theta,
            phi=phi,
            dtheta=dtheta,
            dphi=dphi,
            k_theta=k_theta,
            k_phi=k_phi,
            R=R,
            sqrt_g=sqrt_g,
            G=G,
            r=r,
            r_theta=r_theta,
            r_phi=r_phi,
            area_weights=area_weights,
        )


def make_torus_surface(
    *,
    R0: float,
    a: float,
    n_theta: int,
    n_phi: int,
    dtype=jnp.float64,
) -> TorusSurface:
    """Construct a circular torus surface on a uniform periodic (θ, φ) grid."""
    theta = jnp.linspace(0.0, 2 * jnp.pi, n_theta, endpoint=False, dtype=dtype)
    phi = jnp.linspace(0.0, 2 * jnp.pi, n_phi, endpoint=False, dtype=dtype)
    dtheta = float(2 * jnp.pi / n_theta)
    dphi = float(2 * jnp.pi / n_phi)

    k_theta = make_wavenumbers(n_theta)
    k_phi = make_wavenumbers(n_phi)

    th = theta[:, None]  # (Nθ,1)
    ph = phi[None, :]  # (1,Nφ)

    cos_th = jnp.cos(th)
    sin_th = jnp.sin(th)
    cos_ph = jnp.cos(ph)
    sin_ph = jnp.sin(ph)

    R = (R0 + a * cos_th).astype(dtype)  # (Nθ,1)

    x = R * cos_ph
    y = R * sin_ph
    z = a * sin_th * jnp.ones_like(ph, dtype=dtype)
    r = jnp.stack([x, y, z], axis=-1)

    # Tangents:
    # r_θ = (-a sinθ cosφ, -a sinθ sinφ, a cosθ)
    r_theta = jnp.stack(
        [
            -a * sin_th * cos_ph,
            -a * sin_th * sin_ph,
            a * cos_th * jnp.ones_like(ph, dtype=dtype),
        ],
        axis=-1,
    )

    # r_φ = (-(R0+a cosθ) sinφ, (R0+a cosθ) cosφ, 0)
    r_phi = jnp.stack([-R * sin_ph, R * cos_ph, jnp.zeros_like(x)], axis=-1)

    G = (R**2).astype(dtype)
    sqrt_g = (a * R).astype(dtype)
    area_weights = (sqrt_g * dtheta * dphi) * jnp.ones((n_theta, n_phi), dtype=dtype)

    return TorusSurface(
        R0=float(R0),
        a=float(a),
        theta=theta,
        phi=phi,
        dtheta=dtheta,
        dphi=dphi,
        k_theta=k_theta,
        k_phi=k_phi,
        R=R,
        sqrt_g=sqrt_g,
        G=G,
        r=r,
        r_theta=r_theta,
        r_phi=r_phi,
        area_weights=area_weights,
    )
