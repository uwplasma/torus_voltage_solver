# Model A: electrode-driven surface potential

This model is intended as a differentiable prototype for “voltage sources/sinks on a conducting shell”.

## Unknowns and units

- Surface electric potential: $V(\theta,\phi)$ [V]
- Surface conductivity: $\sigma_s$ [S] (sheet conductivity)
- Surface current density: $\mathbf K(\theta,\phi)$ [A/m]
- Electrode current injection density: $s(\theta,\phi)$ [A/m²]

## Governing equations

We assume a quasi-static surface Ohm’s law:

$$
\mathbf K = -\sigma_s \nabla_s V.
$$

Current continuity on the surface with injected sources/sinks:

$$
\nabla_s\cdot \mathbf K = -s.
$$

Combining yields a Poisson equation on the surface:

$$
-\sigma_s \Delta_s V = s.
$$

### Global constraints on a closed surface

Because the torus surface is closed, the injected current must sum to zero for a steady solution:

$$
\iint s\, dA = 0.
$$

The code enforces this by subtracting the area-mean of $s$ before solving.

The potential has a gauge freedom $V \to V + C$, so the code also fixes the gauge (e.g. by enforcing area-mean $V=0$).

## How electrodes are represented

An electrode $i$ is parameterized by:

- angular position $(\theta_i,\phi_i)$
- injected current $I_i$ [A]

The injection density is modeled as a periodic kernel on the surface:

$$
s(\theta,\phi) = \sum_i I_i\, w_i(\theta,\phi),
$$

where each $w_i$ integrates to $1$ over the surface.

In the code, $w_i$ is a periodic Gaussian in $(\theta,\phi)$ controlled by `sigma_theta` and `sigma_phi`.

## Where this lives in the code

- Electrode deposition: `torus_solver/src/torus_solver/sources.py`
  - `deposit_current_sources(...)`
- Poisson solve: `torus_solver/src/torus_solver/poisson.py`
  - `solve_current_potential(...)`
- Current from potential: `torus_solver/src/torus_solver/poisson.py`
  - `surface_current_from_potential(...)`

:::{warning}
This is a simplified electrostatic model. It is useful for algorithm development, differentiable optimization, and intuition — but it does not model the full Maxwell boundary-value problem of a real conductor with contacts, finite thickness, and electromagnetic coupling.
:::

