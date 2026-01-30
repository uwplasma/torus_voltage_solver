"""JAX-differentiable surface currents on a circular torus.

Core model:
  - A thin conducting torus surface carries a surface current density K driven by
    a scalar electric potential V on the surface: K = -σ_s ∇_s V.
  - Discrete source/sink electrodes inject/extract current on the surface,
    producing a source density s (A/m^2) that satisfies continuity:
        -∇_s^2 V = s / σ_s.
  - The magnetic field is computed from K via the Biot–Savart integral over the
    surface.

This package focuses on a circular torus surface (major radius R0, minor radius a)
and uses spectral derivatives in (θ, φ).
"""

from .torus import TorusSurface, make_torus_surface
from .sources import deposit_current_sources
from .poisson import (
    area_mean,
    laplace_beltrami_torus,
    solve_current_potential,
    surface_current_from_potential,
)
from .biot_savart import biot_savart_surface
from .fields import ideal_toroidal_field, tokamak_like_field
from .fieldline import trace_field_line, trace_field_lines, trace_field_lines_batch

__all__ = [
    "TorusSurface",
    "make_torus_surface",
    "deposit_current_sources",
    "area_mean",
    "laplace_beltrami_torus",
    "solve_current_potential",
    "surface_current_from_potential",
    "biot_savart_surface",
    "ideal_toroidal_field",
    "tokamak_like_field",
    "trace_field_line",
    "trace_field_lines",
    "trace_field_lines_batch",
]
