from __future__ import annotations

import jax.numpy as jnp


def make_wavenumbers(n: int, period: float = 2 * jnp.pi) -> jnp.ndarray:
    """Return Fourier wavenumbers (rad⁻¹) for a periodic grid of length `period`."""
    dx = period / n
    return (2 * jnp.pi) * jnp.fft.fftfreq(n, d=dx)


def spectral_derivative(f: jnp.ndarray, k: jnp.ndarray, *, axis: int) -> jnp.ndarray:
    """Compute ∂f/∂x for a periodic coordinate using FFTs."""
    F = jnp.fft.fft(f, axis=axis)
    shape = [1] * f.ndim
    shape[axis] = k.shape[0]
    kk = k.reshape(shape)
    df = jnp.fft.ifft(1j * kk * F, axis=axis)
    return df.real


def spectral_second_derivative(f: jnp.ndarray, k: jnp.ndarray, *, axis: int) -> jnp.ndarray:
    """Compute ∂²f/∂x² for a periodic coordinate using FFTs."""
    F = jnp.fft.fft(f, axis=axis)
    shape = [1] * f.ndim
    shape[axis] = k.shape[0]
    kk = k.reshape(shape)
    d2f = jnp.fft.ifft(-(kk**2) * F, axis=axis)
    return d2f.real
