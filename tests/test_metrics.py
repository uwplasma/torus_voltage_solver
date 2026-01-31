import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from torus_solver.metrics import bn_over_B, bn_over_B_metrics, weighted_p_norm, weighted_rms


def test_bn_over_B_basic_cases():
    B = jnp.array([[1.0, 0.0, 0.0], [0.0, -3.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float64)
    n = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float64)
    ratio = bn_over_B(B, n)
    np.testing.assert_allclose(np.asarray(ratio), np.array([1.0, -1.0, 0.0]), rtol=0, atol=1e-15)


def test_bn_over_B_metrics_weighted_rms_and_max():
    B = jnp.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=jnp.float64)
    n = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float64)
    w = jnp.array([1.0, 3.0], dtype=jnp.float64)
    ratio, rms, mx = bn_over_B_metrics(B, n, w)
    np.testing.assert_allclose(np.asarray(ratio), np.array([1.0, 1.0]), rtol=0, atol=1e-15)
    assert float(rms) == 1.0
    assert float(mx) == 1.0


def test_weighted_p_norm_matches_rms_at_p2():
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
    w = jnp.array([1.0, 1.0, 2.0], dtype=jnp.float64)
    rms = weighted_rms(x, w)
    p2 = weighted_p_norm(x, w, p=2.0)
    np.testing.assert_allclose(np.asarray(p2), np.asarray(rms), rtol=0, atol=1e-15)

