from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import optax
from jaxopt import LBFGS

from .biot_savart import biot_savart_surface
from .poisson import solve_current_potential, surface_current_from_potential
from .sources import deposit_current_sources
from .torus import TorusSurface


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SourceParams:
    """Electrode locations and injected currents (unconstrained)."""

    theta_src: jnp.ndarray  # (Ns,)
    phi_src: jnp.ndarray  # (Ns,)
    currents_raw: jnp.ndarray  # (Ns,)

    def tree_flatten(self):
        return (self.theta_src, self.phi_src, self.currents_raw), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        theta_src, phi_src, currents_raw = children
        return cls(theta_src=theta_src, phi_src=phi_src, currents_raw=currents_raw)


def enforce_net_zero(currents_raw: jnp.ndarray) -> jnp.ndarray:
    """Project currents onto the Σ I_i = 0 subspace."""
    return currents_raw - jnp.mean(currents_raw)


def forward_B(
    surface: TorusSurface,
    params: SourceParams,
    *,
    eval_points: jnp.ndarray,
    sigma_theta: float,
    sigma_phi: float,
    sigma_s: float = 1.0,
    current_scale: float = 1.0,
    tol: float = 1e-10,
    maxiter: int = 2_000,
    biot_savart_eps: float = 1e-9,
) -> jnp.ndarray:
    """Compute B at `eval_points` from electrode params on the torus surface."""
    _, _, _, K = surface_solution(
        surface,
        params,
        sigma_theta=sigma_theta,
        sigma_phi=sigma_phi,
        sigma_s=sigma_s,
        current_scale=current_scale,
        tol=tol,
        maxiter=maxiter,
    )
    return biot_savart_surface(surface, K, eval_points, eps=biot_savart_eps)


def surface_solution(
    surface: TorusSurface,
    params: SourceParams,
    *,
    sigma_theta: float,
    sigma_phi: float,
    sigma_s: float = 1.0,
    current_scale: float = 1.0,
    tol: float = 1e-10,
    maxiter: int = 2_000,
    use_preconditioner: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Return (currents, source_density, potential, surface_current)."""
    currents = current_scale * enforce_net_zero(params.currents_raw)
    s = deposit_current_sources(
        surface,
        theta_src=params.theta_src,
        phi_src=params.phi_src,
        currents=currents,
        sigma_theta=sigma_theta,
        sigma_phi=sigma_phi,
    )
    V, _ = solve_current_potential(
        surface, s, tol=tol, maxiter=maxiter, use_preconditioner=use_preconditioner
    )
    K = surface_current_from_potential(surface, V, sigma_s=sigma_s)
    return currents, s, V, K


def make_helical_axis_points(
    *,
    R_axis: float,
    n_points: int,
    dtype=jnp.float64,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convenience: points on the torus magnetic axis (major circle) + basis."""
    phi = jnp.linspace(0.0, 2 * jnp.pi, n_points, endpoint=False, dtype=dtype)
    c = jnp.cos(phi)
    s = jnp.sin(phi)
    points = jnp.stack([R_axis * c, R_axis * s, jnp.zeros_like(phi)], axis=-1)
    e_r = jnp.stack([c, s, jnp.zeros_like(phi)], axis=-1)
    e_phi = jnp.stack([-s, c, jnp.zeros_like(phi)], axis=-1)
    e_z = jnp.tile(jnp.array([0.0, 0.0, 1.0], dtype=dtype), (n_points, 1))
    return phi, points, e_r, e_phi, e_z


def optimize_sources(
    surface: TorusSurface,
    *,
    init: SourceParams,
    eval_points: jnp.ndarray,
    B_target: jnp.ndarray,
    B_scale: float = 1.0,
    sigma_theta: float,
    sigma_phi: float,
    n_steps: int,
    lr: float,
    reg_currents: float = 1e-6,
    reg_positions: float = 0.0,
    callback: Callable[[int, Dict[str, float]], None] | None = None,
    return_history: bool = False,
) -> SourceParams | tuple[SourceParams, Dict[str, list[float]]]:
    """Optimize electrode params to match a target B field at given points."""

    def loss_fn(p: SourceParams) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        B = forward_B(
            surface,
            p,
            eval_points=eval_points,
            sigma_theta=sigma_theta,
            sigma_phi=sigma_phi,
        )
        err = (B - B_target) / B_scale
        loss_B = jnp.mean(jnp.sum(err * err, axis=-1))
        currents = enforce_net_zero(p.currents_raw)
        currents_rms = jnp.sqrt(jnp.mean(currents * currents))
        loss_reg = reg_currents * jnp.mean(currents * currents)
        if reg_positions != 0.0:
            loss_reg = loss_reg + reg_positions * (
                jnp.mean(p.theta_src * p.theta_src) + jnp.mean(p.phi_src * p.phi_src)
            )
        loss = loss_B + loss_reg
        aux = {"loss": loss, "loss_B": loss_B, "loss_reg": loss_reg, "currents_rms": currents_rms}
        return loss, aux

    opt = optax.adam(lr)
    opt_state = opt.init(init)

    @jax.jit
    def step(p, s):
        (loss, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p)
        updates, s2 = opt.update(g, s, p)
        p2 = optax.apply_updates(p, updates)
        return p2, s2, aux

    params = init
    state = opt_state
    history: Dict[str, list[float]] = {"loss": [], "loss_B": [], "loss_reg": [], "currents_rms": []}
    for k in range(n_steps):
        params, state, aux = step(params, state)
        if callback is not None:
            callback(
                k,
                {
                    "loss": float(aux["loss"]),
                    "loss_B": float(aux["loss_B"]),
                    "loss_reg": float(aux["loss_reg"]),
                    "currents_rms": float(aux["currents_rms"]),
                },
            )
        if return_history:
            for name in history.keys():
                history[name].append(float(aux[name]))

    if return_history:
        return params, history
    return params


def optimize_sources_lbfgs(
    surface: TorusSurface,
    *,
    init: SourceParams,
    eval_points: jnp.ndarray,
    B_target: jnp.ndarray,
    B_scale: float = 1.0,
    sigma_theta: float,
    sigma_phi: float,
    maxiter: int,
    tol: float = 1e-9,
    reg_currents: float = 1e-6,
    reg_positions: float = 0.0,
    callback: Callable[[int, Dict[str, float]], None] | None = None,
    return_history: bool = False,
) -> SourceParams | tuple[SourceParams, Dict[str, list[float]]]:
    """Optimize electrode params using L-BFGS (often faster than Adam for small problems)."""

    def loss_fn(p: SourceParams) -> tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        B = forward_B(
            surface,
            p,
            eval_points=eval_points,
            sigma_theta=sigma_theta,
            sigma_phi=sigma_phi,
        )
        err = (B - B_target) / B_scale
        loss_B = jnp.mean(jnp.sum(err * err, axis=-1))
        currents = enforce_net_zero(p.currents_raw)
        currents_rms = jnp.sqrt(jnp.mean(currents * currents))
        loss_reg = reg_currents * jnp.mean(currents * currents)
        if reg_positions != 0.0:
            loss_reg = loss_reg + reg_positions * (
                jnp.mean(p.theta_src * p.theta_src) + jnp.mean(p.phi_src * p.phi_src)
            )
        loss = loss_B + loss_reg
        aux = {"loss": loss, "loss_B": loss_B, "loss_reg": loss_reg, "currents_rms": currents_rms}
        return loss, aux

    solver = LBFGS(fun=loss_fn, has_aux=True, maxiter=int(maxiter), tol=float(tol), jit=True)
    params = init
    state = solver.init_state(params)

    history: Dict[str, list[float]] = {"loss": [], "loss_B": [], "loss_reg": [], "currents_rms": []}
    for k in range(int(maxiter)):
        params, state = solver.update(params, state)
        aux = state.aux

        if callback is not None:
            callback(
                k,
                {
                    "loss": float(aux["loss"]),
                    "loss_B": float(aux["loss_B"]),
                    "loss_reg": float(aux["loss_reg"]),
                    "currents_rms": float(aux["currents_rms"]),
                },
            )
        if return_history:
            for name in history.keys():
                history[name].append(float(aux[name]))

        if float(state.error) <= float(tol):
            break

    if return_history:
        return params, history
    return params
