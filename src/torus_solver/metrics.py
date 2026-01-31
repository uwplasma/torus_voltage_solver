from __future__ import annotations

import jax.numpy as jnp


def weighted_mean(x: jnp.ndarray, weights: jnp.ndarray, *, eps: float = 1e-30) -> jnp.ndarray:
    """Weighted mean of a scalar field."""
    x = jnp.asarray(x)
    w = jnp.asarray(weights)
    return jnp.sum(w * x) / (jnp.sum(w) + eps)


def weighted_rms(x: jnp.ndarray, weights: jnp.ndarray, *, eps: float = 1e-30) -> jnp.ndarray:
    """Weighted RMS of a scalar field."""
    x = jnp.asarray(x)
    w = jnp.asarray(weights)
    return jnp.sqrt(weighted_mean(x * x, w, eps=eps))


def weighted_p_norm(x: jnp.ndarray, weights: jnp.ndarray, *, p: float, eps: float = 1e-30) -> jnp.ndarray:
    """Weighted p-norm proxy (p>=2) that interpolates between RMS (p=2) and max (p→∞)."""
    x = jnp.asarray(x)
    w = jnp.asarray(weights)
    mean = weighted_mean(jnp.abs(x) ** p, w, eps=eps)
    return mean ** (1.0 / p)


def bn_over_B(B: jnp.ndarray, normals: jnp.ndarray, *, eps: float = 1e-30) -> jnp.ndarray:
    """Compute the normalized normal field B·n/norm(B)."""
    B = jnp.asarray(B)
    n = jnp.asarray(normals)
    Bn = jnp.sum(B * n, axis=-1)
    Bmag = jnp.linalg.norm(B, axis=-1)
    return Bn / (Bmag + eps)


def bn_over_B_metrics(
    B: jnp.ndarray, normals: jnp.ndarray, weights: jnp.ndarray, *, eps: float = 1e-30
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (Bn_over_B, rms, max_abs) with area weights."""
    ratio = bn_over_B(B, normals, eps=eps)
    rms = weighted_rms(ratio, weights, eps=eps)
    max_abs = jnp.max(jnp.abs(ratio))
    return ratio, rms, max_abs
