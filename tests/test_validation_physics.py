import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from torus_solver import biot_savart_surface, make_torus_surface, surface_current_from_current_potential_with_net_currents
from torus_solver.biot_savart import MU0
from torus_solver.fields import cylindrical_coords, cylindrical_unit_vectors
from torus_solver.metrics import bn_over_B
from torus_solver.targets import circular_torus_target_surface


def test_ampere_law_toroidal_field_from_net_poloidal_current_matches_1_over_R():
    # Current-potential net poloidal current should generate an approximately toroidal 1/R field:
    #   B_phi(R) ≈ μ0 I_pol / (2π R)
    R0 = 3.0
    a = 0.3
    surface = make_torus_surface(R0=R0, a=a, n_theta=96, n_phi=128)

    Ipol = 2.0e6  # [A]
    Phi_sv = jnp.zeros((surface.theta.size, surface.phi.size), dtype=jnp.float64)
    K = surface_current_from_current_potential_with_net_currents(
        surface, Phi_sv, net_poloidal_current_A=Ipol
    )

    R_vals = jnp.array([R0 - 0.10, R0, R0 + 0.10], dtype=jnp.float64)
    pts = jnp.stack([R_vals, jnp.zeros_like(R_vals), jnp.zeros_like(R_vals)], axis=-1)
    B = biot_savart_surface(surface, K, pts, eps=1e-9)

    _, phi, _ = cylindrical_coords(pts)
    _, e_phi, _ = cylindrical_unit_vectors(phi)
    Bphi = jnp.sum(B * e_phi, axis=-1)

    Bphi_expected = float(MU0) * Ipol / (2 * np.pi * R_vals)
    rel = jnp.max(jnp.abs(Bphi - Bphi_expected) / (jnp.abs(Bphi_expected) + 1e-30))
    assert float(rel) < 8e-2


def test_physical_scaling_Bphi_with_R0_for_fixed_Ipol():
    # For fixed I_pol, Ampere's law predicts B_phi(R0) ~ 1/R0.
    Ipol = 2.0e6
    a = 0.25

    def Bphi_at_R0(R0: float) -> float:
        surface = make_torus_surface(R0=R0, a=a, n_theta=64, n_phi=96)
        Phi_sv = jnp.zeros((surface.theta.size, surface.phi.size), dtype=jnp.float64)
        K = surface_current_from_current_potential_with_net_currents(
            surface, Phi_sv, net_poloidal_current_A=Ipol
        )
        pt = jnp.array([[R0, 0.0, 0.0]], dtype=jnp.float64)
        B = biot_savart_surface(surface, K, pt, eps=1e-9)[0]
        # At φ=0, e_phi = +ŷ.
        return float(B[1])

    R0a = 2.5
    R0b = 5.0
    Bphi_a = Bphi_at_R0(R0a)
    Bphi_b = Bphi_at_R0(R0b)
    ratio_num = abs(Bphi_a) / (abs(Bphi_b) + 1e-30)
    ratio_exp = R0b / R0a
    assert abs(ratio_num / ratio_exp - 1.0) < 1.5e-1


def test_convergence_max_bn_over_B_for_symmetric_toroidal_field():
    # For a purely toroidal field, B·n should be zero on any interior torus.
    # Numerical discretization creates small non-toroidal components; these should
    # decrease as the winding-surface resolution increases.
    R0 = 3.0
    a = 0.35
    rho_target = 0.15
    target = circular_torus_target_surface(R0=R0, a=rho_target, n_theta=10, n_phi=14)
    pts = target.xyz.reshape((-1, 3))
    n_hat = target.normals.reshape((-1, 3))

    Ipol = 1.5e6

    def max_bn_over_B(n_theta: int, n_phi: int) -> float:
        surface = make_torus_surface(R0=R0, a=a, n_theta=n_theta, n_phi=n_phi)
        Phi_sv = jnp.zeros((surface.theta.size, surface.phi.size), dtype=jnp.float64)
        K = surface_current_from_current_potential_with_net_currents(
            surface, Phi_sv, net_poloidal_current_A=Ipol
        )
        B = biot_savart_surface(surface, K, pts, eps=1e-9, chunk_size=256)
        ratio = bn_over_B(B, n_hat)
        return float(jnp.max(jnp.abs(ratio)))

    err_lo = max_bn_over_B(24, 32)
    err_hi = max_bn_over_B(64, 96)

    assert err_hi < 0.7 * err_lo
    assert err_hi < 2e-2
