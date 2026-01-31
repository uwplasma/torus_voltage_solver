# References and related work

This project is inspired by and designed to interoperate with common stellarator/toroidal workflows.

## Coil design and current potential methods

- **Current potential representation**: $\mathbf K = \hat{\mathbf n}\times \nabla_s \Phi$ on a surface (used by many winding-surface / inverse magnetostatics approaches).
- **NESCOIL** (Merkel): an influential “inverse magnetostatics” approach for representing coils via a current potential on a coil surface (Nucl. Fusion 27 (1987) 867).
- **REGCOIL** (Landreman): adds Tikhonov-style regularization to avoid pathological growth with resolution and provides a tunable tradeoff between field quality and current magnitude (Nucl. Fusion 57 (2017) 046003, DOI: 10.1088/1741-4326/aa57d4). (See `regcoil-master/` in this workspace.)

Related open-source toolchains in the community include:

- **SIMSOPT**: Python optimization framework for stellarator design workflows (coils, equilibria, objectives). Landreman et al. (JOSS 2021), DOI: 10.21105/joss.03525.
- **FOCUS / FOCUSADD**: direct coil-shape optimization using gradient-based methods. See e.g. Zhu et al. (PPCF 2018), DOI: 10.1088/1361-6587/aab8c2.
- **QUADCOIL**: quadratic-constraint / convex-relaxation coil optimization formulations to enforce engineering constraints more directly. See e.g. Fu et al. (Nucl. Fusion 2025), DOI: 10.1088/1741-4326/ada810.

## MHD equilibrium and target surfaces

- **VMEC**: a widely-used 3D MHD equilibrium code that represents plasma boundaries by Fourier series in $(\theta,\phi)$.
- **Near-axis / quasisymmetry tools** (e.g. “Garren–Boozer” constructions): used to generate quasisymmetric target surfaces and magnetic-axis parameterizations.

## JAX ecosystem

- **JAX**: composable transformations (JIT, vmap, grad) for high-performance differentiable computing.
- **Optax**: gradient-based optimization (Adam, etc).
- **JAXopt**: classical optimization methods (L-BFGS, etc).
- **Equinox**: neural network / PyTree utilities for JAX.

:::{note}
This documentation intentionally emphasizes the mapping between physics equations and code structure. For publication-quality scientific references, consult the upstream projects (REGCOIL, VMEC, pyQSC) and their associated papers.
:::
