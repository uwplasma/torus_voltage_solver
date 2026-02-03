from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from .vmec import VMECBoundary, vmec_boundary_RZ_and_derivatives


@dataclass(frozen=True)
class FitResult:
    """Shift/scale parameters used to fit an (R,Z) surface inside a circular torus."""

    shift_R: float
    scale_rho: float
    rho_max_before_m: float


@dataclass(frozen=True)
class TargetSurface:
    """A target surface sampled on a (theta,phi) grid."""

    theta: jnp.ndarray  # (Nθ,)
    phi: jnp.ndarray  # (Nφ,)
    xyz: jnp.ndarray  # (Nθ,Nφ,3)
    normals: jnp.ndarray  # (Nθ,Nφ,3) outward unit normals
    weights: jnp.ndarray  # (Nθ,Nφ) area weights (|r_theta x r_phi|)
    fit: FitResult


def fit_RZ_surface_into_torus(
    *,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    torus_R0: float,
    torus_a: float,
    fit_margin: float,
) -> FitResult:
    """Compute shift/scale so the (R,Z) surface fits inside the torus minor radius."""
    # Shift so mean(R) matches torus_R0.
    R_mean = jnp.mean(R)
    shift_R = float(torus_R0 - float(R_mean))

    # Scale the radial distance from the torus axis if needed.
    R_shift = R + shift_R
    dR = R_shift - float(torus_R0)
    rho = jnp.sqrt(dR * dR + Z * Z + 1e-30)
    rho_max = float(jnp.max(rho))

    target = float(fit_margin * torus_a)
    scale_rho = 1.0
    if rho_max > target:
        scale_rho = float(target / rho_max)

    return FitResult(shift_R=shift_R, scale_rho=scale_rho, rho_max_before_m=rho_max)


def apply_fit_to_RZ_and_derivatives(
    *,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    R_theta: jnp.ndarray,
    R_phi: jnp.ndarray,
    Z_theta: jnp.ndarray,
    Z_phi: jnp.ndarray,
    torus_R0: float,
    fit: FitResult,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Apply `FitResult` shift/scale to (R,Z) and first derivatives."""
    R_shift = R + float(fit.shift_R)
    dR = R_shift - float(torus_R0)
    R_fit = float(torus_R0) + float(fit.scale_rho) * dR
    Z_fit = float(fit.scale_rho) * Z
    R_theta_fit = float(fit.scale_rho) * R_theta
    R_phi_fit = float(fit.scale_rho) * R_phi
    Z_theta_fit = float(fit.scale_rho) * Z_theta
    Z_phi_fit = float(fit.scale_rho) * Z_phi
    return R_fit, Z_fit, R_theta_fit, R_phi_fit, Z_theta_fit, Z_phi_fit


def RZ_and_derivatives_to_xyz_normals_weights(
    *,
    R: jnp.ndarray,
    Z: jnp.ndarray,
    R_theta: jnp.ndarray,
    R_phi: jnp.ndarray,
    Z_theta: jnp.ndarray,
    Z_phi: jnp.ndarray,
    phi: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Build xyz, unit normals, and area weights from (R,Z) and derivatives."""
    phi2 = jnp.asarray(phi, dtype=jnp.float64)[None, :]
    c = jnp.cos(phi2)
    s = jnp.sin(phi2)

    x = R * c
    y = R * s
    z = Z
    xyz = jnp.stack([x, y, z], axis=-1)

    r_theta_xyz = jnp.stack([R_theta * c, R_theta * s, Z_theta], axis=-1)
    r_phi_xyz = jnp.stack([R_phi * c - R * s, R_phi * s + R * c, Z_phi], axis=-1)
    n_vec = jnp.cross(r_theta_xyz, r_phi_xyz)
    n_norm = jnp.linalg.norm(n_vec, axis=-1)
    n_hat = n_vec / (n_norm[..., None] + 1e-30)
    return xyz, n_hat, n_norm


def vmec_target_surface(
    boundary: VMECBoundary,
    *,
    torus_R0: float,
    torus_a: float,
    fit_margin: float,
    n_theta: int,
    n_phi: int,
    dtype=jnp.float64,
) -> TargetSurface:
    """Sample a VMEC boundary, fit it inside the torus, and return xyz/normals/weights."""
    theta = jnp.linspace(0.0, 2 * jnp.pi, int(n_theta), endpoint=False, dtype=dtype)
    phi = jnp.linspace(0.0, 2 * jnp.pi, int(n_phi), endpoint=False, dtype=dtype)

    R, Z, R_th, R_ph, Z_th, Z_ph = vmec_boundary_RZ_and_derivatives(boundary, theta=theta, phi=phi)
    fit = fit_RZ_surface_into_torus(R=R, Z=Z, torus_R0=torus_R0, torus_a=torus_a, fit_margin=fit_margin)
    R_fit, Z_fit, R_th_fit, R_ph_fit, Z_th_fit, Z_ph_fit = apply_fit_to_RZ_and_derivatives(
        R=R,
        Z=Z,
        R_theta=R_th,
        R_phi=R_ph,
        Z_theta=Z_th,
        Z_phi=Z_ph,
        torus_R0=torus_R0,
        fit=fit,
    )

    xyz, normals, weights = RZ_and_derivatives_to_xyz_normals_weights(
        R=R_fit, Z=Z_fit, R_theta=R_th_fit, R_phi=R_ph_fit, Z_theta=Z_th_fit, Z_phi=Z_ph_fit, phi=phi
    )
    return TargetSurface(theta=theta, phi=phi, xyz=xyz, normals=normals, weights=weights, fit=fit)


def circular_torus_target_surface(
    *,
    R0: float,
    a: float,
    n_theta: int,
    n_phi: int,
    dtype=jnp.float64,
) -> TargetSurface:
    """Return a circular torus target surface (xyz, normals, weights) on a (theta,phi) grid.

    This helper is useful both for:

    - benchmarking/validation: axisymmetry implies B·n=0 on any interior torus for a purely toroidal field
    - GUI/optimization: choosing a simple reference surface inside the winding surface
    """
    theta = jnp.linspace(0.0, 2 * jnp.pi, int(n_theta), endpoint=False, dtype=dtype)
    phi = jnp.linspace(0.0, 2 * jnp.pi, int(n_phi), endpoint=False, dtype=dtype)

    th = theta[:, None]
    ones_phi = jnp.ones((1, int(n_phi)), dtype=dtype)

    R_line = float(R0) + float(a) * jnp.cos(th)  # (Nθ,1)
    Z_line = float(a) * jnp.sin(th)  # (Nθ,1)
    R = R_line * ones_phi  # (Nθ,Nφ)
    Z = Z_line * ones_phi  # (Nθ,Nφ)

    R_theta = (-float(a) * jnp.sin(th)) * ones_phi
    Z_theta = (float(a) * jnp.cos(th)) * ones_phi
    R_phi = jnp.zeros((int(n_theta), int(n_phi)), dtype=dtype)
    Z_phi = jnp.zeros((int(n_theta), int(n_phi)), dtype=dtype)

    xyz, normals, weights = RZ_and_derivatives_to_xyz_normals_weights(
        R=R, Z=Z, R_theta=R_theta, R_phi=R_phi, Z_theta=Z_theta, Z_phi=Z_phi, phi=phi
    )
    fit = FitResult(shift_R=0.0, scale_rho=1.0, rho_max_before_m=float(a))
    return TargetSurface(theta=theta, phi=phi, xyz=xyz, normals=normals, weights=weights, fit=fit)
