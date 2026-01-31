# Roadmap and extension ideas

This project intentionally starts from a “minimal torus” so that validation and differentiation are easy, but it is meant to grow into a research platform for coil/current optimization.

## Near-term goals (high leverage)

- **Convergence studies**: add scripts that sweep $(N_\theta,N_\phi)$ and Fourier mode cutoffs, demonstrating convergence of $B_n/|B|$ and field-line diagnostics.
- **Stronger regression testing**: store small reference metrics for key examples (e.g. final `max|Bn/B|`) and fail CI if they drift unexpectedly.
- **Better diagnostics**:
  - Poincaré sections
  - rotational transform estimates
  - field-line divergence metrics (surface distance over time)

## Geometry generalization

Today, the winding surface is a circular torus with spectral derivatives. A natural next step is to support more general surfaces:

- “deformed tori” (Fourier surfaces) with accurate metric tensors and surface operators
- triangulated surfaces with discrete exterior calculus (DEC) operators

The key design constraint is to keep everything:

- differentiable (or differentiable “enough” for optimization)
- fast under `jax.jit`

## Faster Biot–Savart backends

Biot–Savart evaluation is the dominant cost at high resolution.
Potential directions:

- better batching and `vmap` structure
- low-rank / FFT-based accelerations for special geometries
- fast multipole methods (FMM) or hierarchical treecodes
- GPU-friendly kernels

## Physics extensions

- **Finite thickness / resistive wall models** (time-dependent diffusion of currents)
- **External background fields** beyond idealized $1/R$ models
- **Plasma response / equilibrium coupling** (outside the scope of this repo today, but a key research direction)

## Optimization and constraints

Beyond minimizing $B_n/|B|$, practical coil design often needs constraints:

- smoothness / curvature penalties on $\Phi$ or $\mathbf K$
- symmetry constraints
- bounds on electrode strengths or actuator sparsity
- manufacturability proxies

Because the code is JAX-first, many of these can be added as differentiable penalties or constraints.

## Interop with existing tools

- **REGCOIL comparison**: establish a “same surface, same target, same regularization” benchmark to cross-check against `regcoil-master/`.
- **VMEC I/O**: expand parsing/writing of VMEC-like surfaces and add utilities for field-period handling.
- **Near-axis / pyQSC**: add a helper that converts near-axis surfaces into VMEC Fourier coefficients for direct targeting.

If you are extending the codebase for research, consider opening an issue/PR with:

- the proposed math/model
- a minimal reproducible example
- a validation plan (what analytic scaling, invariants, or external code will you compare against?)

