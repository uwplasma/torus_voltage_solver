# Getting started

## What problem does `torus-solver` solve?

Many toroidal devices (tokamaks, stellarators, and related concepts) can be viewed as an **inverse problem**:

> Choose currents on (or near) a winding surface so that the magnetic field inside the device has desirable properties.

`torus-solver` focuses on a deliberately minimal but powerful setting:

- The **winding surface** is a **circular torus** (major radius `R0`, minor radius `a`).
- The code computes **surface currents** and the resulting magnetic field using **JAX**, so that objectives can be optimized with **automatic differentiation**.

At a high level:

1. Choose a model for the surface current on the winding surface.
2. Compute the magnetic field `B(x)` inside/near the torus using Biot–Savart.
3. Optimize parameters (electrode strengths/locations or current-potential coefficients) to minimize a physics-motivated loss.

The primary “stellarator-style” target used in this repository is a VMEC boundary surface. A common proxy objective is to minimize the **normalized normal field**:

$$
\frac{B\cdot \hat n}{|B|},
$$

on that target surface.

:::{note}
Driving $(B\cdot\hat n)/|B| \to 0$ on a single interior surface is **not sufficient** to guarantee nested magnetic surfaces everywhere inside (integrability is subtle in 3D). However, it is a widely-used and extremely useful ingredient for design workflows and validation.
:::

## Quickstart (run scripts)

From the repo root (`torus_voltage_solver/`):

```bash
pytest -q
python examples/1_simple/tokamak_like_fieldlines.py
python examples/3_advanced/optimize_vmec_surface_Bn.py --model current-potential --vmec-input examples/data/vmec/input.QA_nfp2
python examples/3_advanced/scan_vmec_surface_regularization.py --vmec-input examples/data/vmec/input.QA_nfp2
```

:::{tip}
Most scripts also write ParaView datasets into `figures/<example>/paraview/scene.vtm`.
Open that file in ParaView to explore the 3D configuration interactively.
:::

## Quickstart (minimal Python)

Compute a surface current from a **REGCOIL-like current potential** and evaluate `B` at a few points:

```python
import jax.numpy as jnp

from torus_solver import make_torus_surface
from torus_solver.biot_savart import biot_savart_surface
from torus_solver.current_potential import surface_current_from_current_potential_with_net_currents

surface = make_torus_surface(R0=1.0, a=0.3, n_theta=64, n_phi=64)

# Single-valued part of Phi(θ,φ) (Fourier series, splines, etc.) can go here.
Phi_sv = jnp.zeros((surface.theta.size, surface.phi.size))

# Net poloidal current sets the dominant toroidal field scale via Ampère’s law.
Ipol = 5e6  # [A]

K = surface_current_from_current_potential_with_net_currents(
    surface, Phi_sv, net_poloidal_current_A=Ipol
)

# Evaluate B at a few points near the magnetic axis.
pts = jnp.array([[1.0, 0.0, 0.0], [1.05, 0.0, 0.0], [1.0, 0.0, 0.05]])
B = biot_savart_surface(surface, K, pts)
print(B)
```

## Recommended learning path

If you are new to the physics and numerics, a productive sequence is:

1. Read **Theory → Geometry** and **Theory → Surface operators**.
2. Read **Theory → Electrode model** and run an electrode GUI example.
3. Read **Theory → Current potential** (REGCOIL-like model), then run the VMEC optimization script.
4. Read **Algorithms → Field-line tracing** and reproduce a drift diagnostic.
5. Read **Algorithms → Validation** and learn how to extend the test suite when you add new features.
