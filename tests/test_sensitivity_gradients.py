import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from torus_solver import biot_savart_surface, make_torus_surface, solve_current_potential, surface_current_from_potential
from torus_solver.current_potential import surface_current_from_current_potential_with_net_currents
from torus_solver.sources import deposit_current_sources
from torus_solver.poisson import area_mean, laplace_beltrami_torus


def _finite_diff_grad(f, x: np.ndarray, *, eps: float) -> np.ndarray:
    g = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        xp = x.copy()
        xm = x.copy()
        xp[i] += eps
        xm[i] -= eps
        g[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def _solve_current_potential_exact(surface, b: jnp.ndarray) -> jnp.ndarray:
    """Exact solve for the same operator used by CG, by explicitly forming the matrix.

    This is only used in tests at very small resolution to validate derivatives against
    finite differences without CG early-stopping artifacts.
    """
    b = b - area_mean(surface, b)

    def matvec(x_grid: jnp.ndarray) -> jnp.ndarray:
        return -laplace_beltrami_torus(surface, x_grid) + area_mean(surface, x_grid)

    n_theta = int(surface.theta.size)
    n_phi = int(surface.phi.size)
    N = n_theta * n_phi
    eye = jnp.eye(N, dtype=jnp.float64).reshape((N, n_theta, n_phi))
    # Each row i is A e_i (flattened). Columns are these vectors, so transpose.
    A_rows = jax.vmap(lambda e: matvec(e).reshape((-1,)))(eye)  # (N,N)
    A = A_rows.T
    x = jnp.linalg.solve(A, b.reshape((-1,))).reshape((n_theta, n_phi))
    return x - area_mean(surface, x)


def test_grad_currents_matches_finite_difference_electrode_model_exact_solve():
    # Sensitivity check: autodiff gradient vs finite differences for a small electrode problem.
    # We use an *exact* linear solve for V to avoid non-smoothness from CG stopping criteria.
    # NOTE: Use odd (n_theta, n_phi) to avoid the Nyquist-mode nullspace artifact that can
    # occur with first-derivative-based operators on even grids, which would make the
    # exact matrix solve ill-conditioned.
    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=21, n_phi=21)

    theta_src = jnp.array([0.2, 1.2, -0.7], dtype=jnp.float64)
    phi_src = jnp.array([0.1, -2.1, 0.9], dtype=jnp.float64)
    sigma_theta = 0.35
    sigma_phi = 0.35
    sigma_s = 1.7
    current_scale = 1e6  # [A] per unit currents_raw (avoid tiny B that breaks finite differences)

    pts = jnp.array([[surface.R0, 0.0, 0.0], [surface.R0 + 0.1, 0.0, 0.05]], dtype=jnp.float64)

    def loss(currents_raw: jnp.ndarray) -> jnp.ndarray:
        # Enforce net-zero like the main code paths do.
        currents = current_scale * (currents_raw - jnp.mean(currents_raw))
        s = deposit_current_sources(
            surface,
            theta_src=theta_src,
            phi_src=phi_src,
            currents=currents,
            sigma_theta=sigma_theta,
            sigma_phi=sigma_phi,
        )
        V = _solve_current_potential_exact(surface, s / sigma_s)
        K = surface_current_from_potential(surface, V, sigma_s=sigma_s)
        B = biot_savart_surface(surface, K, pts, eps=1e-9)
        return jnp.sum(B * B)

    x0 = np.array([0.8, -0.2, -0.1], dtype=float)
    g_ad = np.asarray(jax.grad(lambda x: loss(jnp.asarray(x, dtype=jnp.float64)))(x0))
    g_fd = _finite_diff_grad(lambda x: float(loss(jnp.asarray(x, dtype=jnp.float64))), x0, eps=1e-6)

    denom = np.linalg.norm(g_fd) + 1e-30
    rel = np.linalg.norm(g_ad - g_fd) / denom
    assert rel < 2e-3


def test_solve_current_potential_cg_matches_exact_on_small_problem():
    # Use odd resolution to avoid the even-grid Nyquist nullspace artifact for the
    # exact matrix solve path.
    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=15, n_phi=15)
    key = jax.random.key(0)
    b = jax.random.normal(key, (surface.theta.size, surface.phi.size), dtype=jnp.float64)
    b = b - area_mean(surface, b)

    V_exact = _solve_current_potential_exact(surface, b)
    V_cg, info = solve_current_potential(surface, b, tol=1e-12, maxiter=5000)
    # `info` type can vary across JAX versions; the key requirement is that the solution
    # matches a direct solve at tight tolerance for this small case.
    np.testing.assert_allclose(np.asarray(V_cg), np.asarray(V_exact), rtol=0, atol=5e-10)


def test_grad_phi_coeff_matches_finite_difference_current_potential():
    # Sensitivity check: autodiff gradient vs finite differences for a simple current-potential coefficient.
    surface = make_torus_surface(R0=3.0, a=0.6, n_theta=24, n_phi=24)

    # Parameterize Phi_sv as alpha * cos(theta) (single-valued), plus a net poloidal current.
    Ipol = 1.0e6
    pts = jnp.array([[surface.R0, 0.0, 0.0], [surface.R0, 0.0, 0.2]], dtype=jnp.float64)

    def loss(alpha: jnp.ndarray) -> jnp.ndarray:
        Phi_sv = (alpha * jnp.cos(surface.theta)[:, None]) * jnp.ones((1, surface.phi.size), dtype=jnp.float64)
        K = surface_current_from_current_potential_with_net_currents(
            surface, Phi_sv, net_poloidal_current_A=Ipol
        )
        B = biot_savart_surface(surface, K, pts, eps=1e-9)
        # Use a simple scalar loss that depends on alpha.
        return jnp.sum(B[:, 2] * B[:, 2])

    a0 = 0.3
    g_ad = float(jax.grad(lambda a: loss(jnp.asarray(a, dtype=jnp.float64)))(a0))
    eps = 1e-6
    g_fd = (float(loss(a0 + eps)) - float(loss(a0 - eps))) / (2 * eps)

    denom = abs(g_fd) + 1e-30
    rel = abs(g_ad - g_fd) / denom
    assert rel < 2e-3
