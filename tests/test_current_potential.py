import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
from pathlib import Path

from torus_solver import make_torus_surface, surface_current_from_current_potential_with_net_currents


def test_current_potential_secular_term_has_expected_max_K():
    R0 = 6.5
    a = 4.0
    surface = make_torus_surface(R0=R0, a=a, n_theta=32, n_phi=32)

    Ipol = 1.0e6
    Phi_sv = jnp.zeros((surface.theta.size, surface.phi.size), dtype=jnp.float64)
    K = surface_current_from_current_potential_with_net_currents(surface, Phi_sv, net_poloidal_current_A=Ipol)
    Kmag = jnp.linalg.norm(K, axis=-1)

    # For Phi = Ipol * φ/(2π), we get |K| = |Ipol|/(2π R(θ)).
    expected_max = Ipol / (2 * np.pi * (R0 - a))
    maxK = float(jnp.max(Kmag))
    assert np.isclose(maxK, expected_max, rtol=0, atol=2e-10)


def test_current_potential_net_toroidal_current_has_expected_constant_K():
    R0 = 6.5
    a = 4.0
    surface = make_torus_surface(R0=R0, a=a, n_theta=48, n_phi=48)

    Itor = 2.5e5
    Phi_sv = jnp.zeros((surface.theta.size, surface.phi.size), dtype=jnp.float64)
    K = surface_current_from_current_potential_with_net_currents(surface, Phi_sv, net_toroidal_current_A=Itor)
    Kmag = jnp.linalg.norm(K, axis=-1)

    # For Phi = Itor * θ/(2π), we get |K| = |Itor|/(2π a) (independent of θ, φ).
    expected = Itor / (2 * np.pi * a)
    maxK = float(jnp.max(Kmag))
    minK = float(jnp.min(Kmag))
    assert np.isclose(maxK, expected, rtol=0, atol=2e-10)
    assert np.isclose(minK, expected, rtol=0, atol=2e-10)


def test_regcoil_axisymmetry_sanitytest_max_K_matches_current_potential():
    """Cross-check against the Fortran REGCOIL axisymmetry sanity test output."""
    try:
        import netCDF4  # type: ignore
    except Exception:  # pragma: no cover
        import pytest

        pytest.skip("netCDF4 not available")

    root = Path(__file__).resolve().parents[2]
    nc = (
        root
        / "regcoil-master"
        / "examples"
        / "axisymmetrySanityTest_Laplace_Beltrami_regularization"
        / "regcoil_out.axisymmetrySanityTest_Laplace_Beltrami_regularization.nc"
    )
    if not nc.exists():  # pragma: no cover
        import pytest

        pytest.skip(f"REGCOIL output not found at {nc}")

    with netCDF4.Dataset(nc) as ds:
        R0 = float(ds.variables["R0_coil"][:])
        a = float(ds.variables["a_coil"][:])
        Ipol = float(ds.variables["net_poloidal_current_Amperes"][:])
        maxK_regcoil = float(ds.variables["max_K"][:][0])

    surface = make_torus_surface(R0=R0, a=a, n_theta=64, n_phi=64)
    Phi_sv = jnp.zeros((surface.theta.size, surface.phi.size), dtype=jnp.float64)
    K = surface_current_from_current_potential_with_net_currents(surface, Phi_sv, net_poloidal_current_A=Ipol)
    maxK = float(jnp.max(jnp.linalg.norm(K, axis=-1)))

    assert np.isclose(maxK, maxK_regcoil, rtol=1e-12, atol=0.0)
