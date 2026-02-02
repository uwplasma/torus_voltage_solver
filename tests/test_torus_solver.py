import jax

# Enable 64-bit for tighter analytic comparisons.
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from torus_solver import (
    biot_savart_surface,
    ideal_toroidal_field,
    laplace_beltrami_torus,
    make_torus_surface,
    solve_current_potential,
    surface_current_from_potential,
    tokamak_like_field,
)
from torus_solver.fieldline import trace_field_line
from torus_solver.fieldline import trace_field_lines, trace_field_lines_batch
from torus_solver.fields import cylindrical_coords, cylindrical_unit_vectors, toroidal_poloidal_coords
from torus_solver.optimize import SourceParams, forward_B
from torus_solver.sources import wrap_angle


MU0 = 4e-7 * np.pi


def test_torus_geometry_metric_identities():
    surface = make_torus_surface(R0=3.0, a=1.2, n_theta=32, n_phi=40)
    r_theta = surface.r_theta
    r_phi = surface.r_phi

    E = jnp.sum(r_theta * r_theta, axis=-1)
    F = jnp.sum(r_theta * r_phi, axis=-1)
    G = jnp.sum(r_phi * r_phi, axis=-1)

    assert np.allclose(np.asarray(E), surface.a**2, rtol=0, atol=1e-12)
    assert np.allclose(np.asarray(F), 0.0, rtol=0, atol=1e-12)
    assert np.allclose(np.asarray(G), np.asarray(surface.G), rtol=0, atol=1e-12)

    dA_from_cross = jnp.linalg.norm(jnp.cross(r_theta, r_phi), axis=-1)
    assert np.allclose(
        np.asarray(dA_from_cross), np.asarray(surface.sqrt_g), rtol=0, atol=1e-12
    )


def test_laplace_beltrami_phi_mode_matches_analytic():
    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=48, n_phi=64)

    m = 3
    V = jnp.sin(m * surface.phi)[None, :] * jnp.ones((surface.theta.size, 1))
    lap_num = laplace_beltrami_torus(surface, V)
    lap_exact = -(m * m) * V / surface.G

    err = jnp.max(jnp.abs(lap_num - lap_exact))
    assert float(err) < 2e-10


def test_poisson_solve_inverts_minus_laplacian():
    surface = make_torus_surface(R0=2.6, a=0.9, n_theta=48, n_phi=64)
    V_true = jnp.sin(2.0 * surface.phi)[None, :] * jnp.ones((surface.theta.size, 1))
    s = -laplace_beltrami_torus(surface, V_true)

    V_sol, _ = solve_current_potential(surface, s, tol=1e-12, maxiter=2000)
    err = jnp.max(jnp.abs(V_sol - V_true))
    assert float(err) < 5e-10


def _B_loop_axis(I: float, R: float, z: np.ndarray) -> np.ndarray:
    """Analytic Bz on the symmetry axis of a circular current loop."""
    z = np.asarray(z, dtype=float)
    return MU0 * I * R * R / (2.0 * np.power(R * R + z * z, 1.5))


def test_biot_savart_matches_circular_loop_on_axis():
    # Thin torus so a narrow band near θ=0 approximates a single loop well.
    R0 = 3.0
    a = 0.2
    surface = make_torus_surface(R0=R0, a=a, n_theta=96, n_phi=128)

    I = 5.0
    sigma_theta = 0.18
    dth = wrap_angle(surface.theta)  # centered at θ=0
    g = jnp.exp(-0.5 * (dth / sigma_theta) ** 2)
    g = g / (jnp.sum(g) * surface.dtheta)  # ∫ g dθ = 1

    # K points along +φ, with ∫ Kφ a dθ = I
    Kphi = (I / a) * g  # (Nθ,)
    e_phi = surface.r_phi / jnp.sqrt(surface.G)[..., None]
    K = Kphi[:, None, None] * e_phi

    z = jnp.array([0.0, 0.5, 1.0, 2.0], dtype=jnp.float64)
    pts = jnp.stack([jnp.zeros_like(z), jnp.zeros_like(z), z], axis=-1)
    B = biot_savart_surface(surface, K, pts, eps=1e-9)

    R_loop = R0 + a
    Bz_exact = _B_loop_axis(I, R_loop, np.asarray(z))
    Bz_num = np.asarray(B[:, 2])

    rel = np.max(np.abs(Bz_num - Bz_exact) / (np.abs(Bz_exact) + 1e-30))
    assert rel < 2e-2


def test_forward_model_is_differentiable():
    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=24, n_phi=24)

    # A few axis points + a simple target.
    phi = jnp.linspace(0.0, 2 * jnp.pi, 8, endpoint=False)
    pts = jnp.stack([surface.R0 * jnp.cos(phi), surface.R0 * jnp.sin(phi), 0.0 * phi], axis=-1)
    B_target = jnp.zeros_like(pts)

    params = SourceParams(
        theta_src=jnp.array([0.3, 1.2, -0.7]),
        phi_src=jnp.array([0.1, -1.5, 2.0]),
        currents_raw=jnp.array([0.2, -0.1, 0.05]),
    )

    def loss(p: SourceParams) -> jnp.ndarray:
        B = forward_B(surface, p, eval_points=pts, sigma_theta=0.35, sigma_phi=0.35)
        return jnp.mean(jnp.sum((B - B_target) ** 2, axis=-1))

    g = jax.grad(loss)(params)
    assert np.all(np.isfinite(np.asarray(g.theta_src)))
    assert np.all(np.isfinite(np.asarray(g.phi_src)))
    assert np.all(np.isfinite(np.asarray(g.currents_raw)))


def test_regression_fixed_sources_fixed_B():
    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=32, n_phi=32)
    theta_src = jnp.array([0.1, 1.7, -0.6])
    phi_src = jnp.array([0.2, -2.1, 0.9])
    currents = jnp.array([1.0, -0.4, -0.6])

    from torus_solver import deposit_current_sources

    s = deposit_current_sources(
        surface,
        theta_src=theta_src,
        phi_src=phi_src,
        currents=currents,
        sigma_theta=0.25,
        sigma_phi=0.25,
    )
    V, _ = solve_current_potential(surface, s, tol=1e-12, maxiter=2000)
    K = surface_current_from_potential(surface, V)

    pts = jnp.array(
        [
            [surface.R0, 0.0, 0.0],
            [surface.R0 / jnp.sqrt(2.0), surface.R0 / jnp.sqrt(2.0), 0.0],
            [surface.R0, 0.0, 0.3],
        ],
        dtype=jnp.float64,
    )
    B = biot_savart_surface(surface, K, pts, eps=1e-9)

    B_expected = np.array(
        [
            [-1.03022524e-09, 6.02817441e-10, -4.15215991e-09],
            [-6.11063196e-10, 1.07112231e-09, -4.36145602e-10],
            [7.61435009e-10, -1.35950245e-08, -3.90631806e-09],
        ],
        dtype=float,
    )
    np.testing.assert_allclose(np.asarray(B), B_expected, rtol=0, atol=2e-12)


def test_ideal_toroidal_field_has_1_over_R_scaling():
    R0 = 3.0
    B0 = 2.5
    phi = jnp.array([0.0, 0.7, -1.2])
    R = jnp.array([2.0, 3.0, 6.0])
    Z = jnp.array([0.1, -0.2, 0.3])
    pts = jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], axis=-1)

    B = ideal_toroidal_field(pts, B0=B0, R0=R0)
    _, phi2, _ = cylindrical_coords(pts)
    _, e_phi, _ = cylindrical_unit_vectors(phi2)
    Bphi = jnp.sum(B * e_phi, axis=-1)

    expected = B0 * R0 / R
    np.testing.assert_allclose(np.asarray(Bphi), np.asarray(expected), rtol=0, atol=5e-12)


def test_tokamak_like_field_has_1_over_R_toroidal_and_poloidal_components():
    R0 = 3.0
    Btor0 = 1.7
    Bpol0 = 0.23
    phi = jnp.array([0.2, -0.5, 1.1])
    R = jnp.array([2.5, 3.5, 5.0])
    Z = jnp.array([0.3, -0.1, 0.25])
    pts = jnp.stack([R * jnp.cos(phi), R * jnp.sin(phi), Z], axis=-1)

    B = tokamak_like_field(pts, B_tor0=Btor0, B_pol0=Bpol0, R0=R0)
    R2, phi2, *_rest, e_rho, e_theta = toroidal_poloidal_coords(pts, R0=R0)
    del _rest, e_rho

    _, e_phi, _ = cylindrical_unit_vectors(phi2)
    Bphi = jnp.sum(B * e_phi, axis=-1)
    Btheta = jnp.sum(B * e_theta, axis=-1)

    expected_scale = R0 / R2
    np.testing.assert_allclose(np.asarray(Bphi), np.asarray(Btor0 * expected_scale), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(Btheta), np.asarray(Bpol0 * expected_scale), rtol=0, atol=1e-12)


def test_trace_field_line_pure_toroidal_is_circle():
    R0 = 3.0
    B0 = 1.0
    R = 3.4
    z0 = 0.25
    r0 = jnp.array([R, 0.0, z0], dtype=jnp.float64)

    def B_fn(xyz):
        return ideal_toroidal_field(xyz, B0=B0, R0=R0)

    step = 0.02
    n_steps = int(np.round((2 * np.pi * R) / step))
    pts = trace_field_line(B_fn, r0, step_size=step, n_steps=n_steps, normalize=True)

    R_tr, _, Z_tr = cylindrical_coords(pts)
    assert float(jnp.max(jnp.abs(R_tr - R))) < 5e-3
    assert float(jnp.max(jnp.abs(Z_tr - z0))) < 5e-3

    # After ~one full toroidal turn, we should be back near the start.
    err = jnp.linalg.norm(pts[-1] - r0)
    assert float(err) < 1e-2


def test_trace_field_line_tokamak_has_poloidal_motion():
    R0 = 3.0
    r0 = jnp.array([R0 + 0.4, 0.0, 0.0], dtype=jnp.float64)

    def B_fn(xyz):
        return tokamak_like_field(xyz, B_tor0=1.0, B_pol0=0.2, R0=R0)

    pts = trace_field_line(B_fn, r0, step_size=0.02, n_steps=2000, normalize=True)
    _, _, Z = cylindrical_coords(pts)
    assert float(jnp.max(Z) - jnp.min(Z)) > 1e-2


def test_biot_savart_uniform_poloidal_current_sheet_matches_ideal_toroidal():
    R0 = 3.0
    a = 0.3
    surface = make_torus_surface(R0=R0, a=a, n_theta=96, n_phi=128)

    B0 = 1e-3  # T at R=R0
    Ktheta = -B0 / MU0  # A/m (sign chosen to match +e_phi at phi=0)
    e_theta = surface.r_theta / surface.a
    K = Ktheta * e_theta

    R_vals = jnp.array([R0 - 0.15, R0, R0 + 0.15], dtype=jnp.float64)
    pts = jnp.stack([R_vals, jnp.zeros_like(R_vals), jnp.zeros_like(R_vals)], axis=-1)

    B_bs = biot_savart_surface(surface, K, pts, eps=1e-9)
    B_an = ideal_toroidal_field(pts, B0=B0, R0=R0)

    # Compare toroidal component at φ=0 (e_phi = +ŷ).
    Bphi_bs = B_bs[:, 1]
    Bphi_an = B_an[:, 1]
    rel = jnp.max(jnp.abs(Bphi_bs - Bphi_an) / (jnp.abs(Bphi_an) + 1e-30))
    assert float(rel) < 5e-2


def test_biot_savart_chunking_matches_direct():
    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=16, n_phi=16)
    key = jax.random.key(0)
    K = jax.random.normal(key, surface.r.shape, dtype=jnp.float64) * 1e3
    pts = jax.random.normal(jax.random.fold_in(key, 1), (300, 3), dtype=jnp.float64)

    B_direct = biot_savart_surface(surface, K, pts, eps=1e-9, chunk_size=None)
    B_chunk = biot_savart_surface(surface, K, pts, eps=1e-9, chunk_size=64)
    np.testing.assert_allclose(np.asarray(B_chunk), np.asarray(B_direct), rtol=0, atol=1e-12)


def test_trace_field_lines_batch_matches_vmap():
    R0 = 3.0
    B0 = 1.0
    theta_seed = jnp.linspace(0.0, 2 * jnp.pi, 8, endpoint=False)
    rho = 0.4
    R_seed = R0 + rho * jnp.cos(theta_seed)
    Z_seed = rho * jnp.sin(theta_seed)
    seeds = jnp.stack([R_seed, jnp.zeros_like(R_seed), Z_seed], axis=-1)

    def B_fn(xyz):
        return ideal_toroidal_field(xyz, B0=B0, R0=R0)

    traj_vmap = trace_field_lines(B_fn, seeds, step_size=0.05, n_steps=50, normalize=True)
    traj_batch = trace_field_lines_batch(B_fn, seeds, step_size=0.05, n_steps=50, normalize=True)
    traj_batch_vmap_order = jnp.transpose(traj_batch, (1, 0, 2))
    np.testing.assert_allclose(
        np.asarray(traj_batch_vmap_order), np.asarray(traj_vmap), rtol=0, atol=2e-12
    )


def test_electrode_model_matches_documented_sigma_poisson_scaling():
    # For the electrode model, the governing equation is:
    #   -sigma_s * Δ_s V = s,   K = -sigma_s ∇_s V
    # For uniform sigma_s, K should be invariant to sigma_s, while V rescales ~ 1/sigma_s.
    surface = make_torus_surface(R0=3.0, a=1.0, n_theta=48, n_phi=64)

    theta_src = jnp.array([0.3, 1.1, -0.7, 2.0])
    phi_src = jnp.array([0.2, -1.5, 0.9, 2.2])
    currents = jnp.array([1.0, -0.4, -0.5, -0.1])  # net ~0 already

    from torus_solver import deposit_current_sources

    s = deposit_current_sources(
        surface,
        theta_src=theta_src,
        phi_src=phi_src,
        currents=currents,
        sigma_theta=0.25,
        sigma_phi=0.25,
    )

    sigma1 = 0.5
    sigma2 = 2.0
    V1, _ = solve_current_potential(surface, s / sigma1, tol=1e-12, maxiter=2000)
    V2, _ = solve_current_potential(surface, s / sigma2, tol=1e-12, maxiter=2000)
    K1 = surface_current_from_potential(surface, V1, sigma_s=sigma1)
    K2 = surface_current_from_potential(surface, V2, sigma_s=sigma2)

    # K should match.
    np.testing.assert_allclose(np.asarray(K1), np.asarray(K2), rtol=0, atol=2e-10)
    # V should scale like 1/sigma: sigma*V should be invariant (up to gauge).
    np.testing.assert_allclose(
        np.asarray(V1) * sigma1,
        np.asarray(V2) * sigma2,
        rtol=0,
        atol=2e-10,
    )
