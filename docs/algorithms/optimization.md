# Optimization

This repository contains two main optimization “stories”:

1. **Electrode optimization**: optimize discrete source/sink currents (and optionally positions) that drive a surface potential solve.
2. **Current-potential optimization**: optimize Fourier coefficients of a REGCOIL-like current potential $\Phi$.

Both are differentiable end-to-end with JAX.

## Objectives: always use normalized $B_n/|B|$

When targeting a surface $S$ with unit normal $\hat{\mathbf n}$, define:

$$
\mathrm{BnOverB}(\theta,\phi) = \frac{\mathbf B(\theta,\phi)\cdot \hat{\mathbf n}(\theta,\phi)}{|\mathbf B(\theta,\phi)|}.
$$

Most scripts report both:

- area-weighted `rms(Bn/B)`
- `max|Bn/B|`

and aim for `max|Bn/B| < 5e-3` (0.5%).

## Current-potential model (REGCOIL-like)

`examples/optimize_vmec_surface_Bn.py --model current-potential` solves:

- parameterization: $\Phi_{\mathrm{sv}}$ as a Fourier series in $(\theta,\phi)$
- net currents: $(I_\mathrm{pol}, I_\mathrm{tor})$
- objective: weighted $p$-norm (default $p=8$) of $\mathrm{BnOverB}$ plus regularization on $\langle|K|^2\rangle$

### Why a $p$-norm?

The $p$-norm interpolates between RMS ($p=2$) and max-norm ($p\to\infty$):

$$
\|x\|_{p,w} = \left(\frac{\sum_i w_i |x_i|^p}{\sum_i w_i}\right)^{1/p}.
$$

This is a smooth proxy for “make the worst-case points small”.

### Mode aliasing and resolution

If the winding-surface grid has $N_\theta\times N_\phi$ points and the device has $N_\mathrm{fp}$ field periods, then toroidal Fourier modes are effectively limited by:

$$
|n| N_\mathrm{fp} \le \frac{N_\phi}{2}-1,
$$

and poloidal modes by:

$$
m \le \frac{N_\theta}{2}-1.
$$

The VMEC optimization script automatically clips requested modes to avoid aliasing.

## Electrode model

The electrode model is conceptually closer to “voltage/current injection hardware”, but it is typically harder to hit stringent $B_n/|B|$ targets than a flexible current-potential representation.

It is still valuable for:

- building intuition
- prototyping GUI workflows
- exploring regularization and constraints on discrete drivers

In this repository:

- `torus_solver.optimize.optimize_sources(...)` uses Adam (Optax).
- `torus_solver.optimize.optimize_sources_lbfgs(...)` uses L-BFGS (JAXopt), which can converge in fewer steps for small/medium problems.
- `examples/optimize_helical_field.py --optimizer lbfgs` demonstrates the L-BFGS path.

## Practical tuning knobs

If an optimization stalls or gives poor results, the most common levers are:

- **Fit margin**: scaling the target surface deeper inside the winding surface can dramatically improve the attainable $B_n/|B|$ and reduce field-line drift.
- **Resolution**: increase `--surf-n-theta/--surf-n-phi` and check convergence.
- **Degrees of freedom**:
  - increase Fourier modes in the current-potential model
  - increase number of electrodes and/or optimize positions in the electrode model
- **Regularization**: adjust `reg_K` (current-potential) or `reg_currents` (electrodes).

## Regularization scan (L-curve) workflow

In current-potential coil design literature (including REGCOIL-style formulations), it is common to scan
the regularization weight that penalizes current magnitude and then choose a compromise point on a tradeoff
curve (often called an “L-curve”):

- smaller regularization → smaller `Bn/B`, but larger `|K|`
- larger regularization → smaller `|K|`, but worse `Bn/B`

`examples/scan_vmec_surface_regularization.py` performs this scan in a way that is friendly to research:

- runs a series of optimizations for a geometric sequence of `reg_K`
- uses continuation / warm-starting (each run initializes from the previous result)
- produces a log–log tradeoff plot of `max|Bn/B|` and `rms(Bn/B)` versus `K_rms`
