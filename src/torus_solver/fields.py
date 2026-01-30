from __future__ import annotations

import jax.numpy as jnp


def cylindrical_coords(xyz: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert Cartesian points (...,3) -> cylindrical (R, φ, Z)."""
    xyz = jnp.asarray(xyz)
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    R = jnp.sqrt(x * x + y * y)
    phi = jnp.arctan2(y, x)
    return R, phi, z


def cylindrical_unit_vectors(phi: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Unit vectors (e_R, e_φ, e_Z) for cylindrical angle φ."""
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    e_R = jnp.stack([c, s, jnp.zeros_like(c)], axis=-1)
    e_phi = jnp.stack([-s, c, jnp.zeros_like(c)], axis=-1)
    e_Z = jnp.stack([jnp.zeros_like(c), jnp.zeros_like(c), jnp.ones_like(c)], axis=-1)
    return e_R, e_phi, e_Z


def toroidal_poloidal_coords(
    xyz: jnp.ndarray, *, R0: float, eps: float = 1e-12
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return toroidal (R, φ, Z) and local poloidal (ρ, θ, e_ρ, e_θ).

    The local poloidal coordinates are defined in the (R,Z) plane relative to the
    circle R=R0, Z=0:
      ρ = sqrt((R-R0)^2 + Z^2)
      θ = atan2(Z, R-R0)
      e_ρ = cosθ e_R + sinθ e_Z
      e_θ = -sinθ e_R + cosθ e_Z
    """
    R, phi, Z = cylindrical_coords(xyz)
    e_R, _, e_Z = cylindrical_unit_vectors(phi)

    dR = R - R0
    rho = jnp.sqrt(dR * dR + Z * Z + eps * eps)
    theta = jnp.arctan2(Z, dR + 0.0)

    c = jnp.cos(theta)
    s = jnp.sin(theta)
    e_rho = c[..., None] * e_R + s[..., None] * e_Z
    e_theta = (-s)[..., None] * e_R + c[..., None] * e_Z
    return R, phi, Z, rho, theta, e_rho, e_theta


def ideal_toroidal_field(
    xyz: jnp.ndarray, *, B0: float, R0: float, eps: float = 1e-12
) -> jnp.ndarray:
    """Ideal axisymmetric toroidal field: B = B0 * (R0/R) e_φ."""
    R, phi, _ = cylindrical_coords(xyz)
    _, e_phi, _ = cylindrical_unit_vectors(phi)
    scale = B0 * R0 / (R + eps)
    return scale[..., None] * e_phi


def tokamak_like_field(
    xyz: jnp.ndarray, *, B_tor0: float, B_pol0: float, R0: float, eps: float = 1e-12
) -> jnp.ndarray:
    """Toy tokamak-like field with 1/R toroidal + poloidal components.

    B = (B_tor0 * R0/R) e_φ + (B_pol0 * R0/R) e_θ
    """
    R, phi, Z, rho, theta, e_rho, e_theta = toroidal_poloidal_coords(xyz, R0=R0, eps=eps)
    del Z, rho, theta, e_rho  # unused, but helpful for debugging during development.
    _, e_phi, _ = cylindrical_unit_vectors(phi)
    scale = R0 / (R + eps)
    return (B_tor0 * scale)[..., None] * e_phi + (B_pol0 * scale)[..., None] * e_theta

