import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from torus_solver import make_torus_surface
from torus_solver.current_potential import surface_current_from_current_potential
from torus_solver.surface_ops import surface_divergence_torus


def test_surface_divergence_current_potential_is_small():
    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=64, n_phi=64)

    th = surface.theta[:, None]
    ph = surface.phi[None, :]
    Phi = jnp.sin(2.0 * th + 3.0 * ph) + 0.3 * jnp.cos(5.0 * th - 2.0 * ph)

    K = surface_current_from_current_potential(surface, Phi)
    div = surface_divergence_torus(surface, K)

    # The discrete divergence is not expected to be *exactly* zero, but should be tiny.
    assert float(jnp.max(jnp.abs(div))) < 5e-10

