# Glossary and conventions

This page is a quick reference for the symbols and conventions used throughout the docs and code.

## Geometry

- **Winding surface**: The surface on which currents live. In this project it is always a **circular torus**.
- **Target surface**: A surface inside the winding surface on which we evaluate objectives such as $B_n/|B|$ (e.g. a VMEC plasma boundary scaled inward).
- $R_0$: Major radius of the circular torus [m].
- $a$: Minor radius of the circular torus [m].
- $(\theta,\phi)$: Poloidal and toroidal angles on $[0,2\pi)$.
- $\mathbf r(\theta,\phi)$: Surface embedding in $\mathbb{R}^3$.
- $\hat{\mathbf n}$: Outward unit normal of a surface.

## Surface operators

- $\nabla_s$: Surface gradient (tangential gradient).
- $\nabla_s\cdot$: Surface divergence.
- $\nabla_s^2$: Laplace–Beltrami operator.
- $\sqrt{g}$: Surface Jacobian factor (area element density) so that $dA=\sqrt{g}\,d\theta\,d\phi$.

## Currents, potentials, and sources

- $\mathbf K(\theta,\phi)$: Surface current density [A/m] on the winding surface.
- $V(\theta,\phi)$: Electrostatic potential on the surface [V] used in the electrode-driven model.
- $\sigma_s$: Surface conductivity [S] (a model parameter; can be absorbed into scalings).
- $s(\theta,\phi)$: Injected/extracted surface source density [A/m$^2$] representing discrete electrodes or distributed sources/sinks.

### Model A (electrodes)

- Constitutive law: $\mathbf K = -\sigma_s \nabla_s V$.
- Continuity: $-\nabla_s^2 V = s/\sigma_s$.

### Model B (current potential / REGCOIL-like)

- $\Phi(\theta,\phi)$: Current potential [A] such that $\mathbf K = \hat{\mathbf n}\times\nabla_s \Phi$.
- $\Phi_{\mathrm{sv}}$: Single-valued (periodic) part of $\Phi$.
- $I_{\mathrm{pol}}$: Net poloidal current [A] (dominant control of the toroidal $1/R$ field scale).
- $I_{\mathrm{tor}}$: Net toroidal current [A] (dominant control of the poloidal field scale).

## Magnetic fields and objectives

- $\mathbf B(\mathbf x)$: Magnetic field [T] in vacuum from Biot–Savart.
- $B_n = \mathbf B\cdot\hat{\mathbf n}$: Normal component on a surface.
- $|B| = \|\mathbf B\|$: Magnitude of the magnetic field.
- **Normalized normal field**: $B_n/|B|$ (dimensionless). This is the quantity reported and optimized throughout the repository.

## VMEC conventions (stellarator symmetry case)

VMEC input files represent the boundary by Fourier series:

$$
R(\theta,\phi)=\sum_{m,n} \mathrm{RBC}_{n,m}\cos(m\theta-nN_{\mathrm{fp}}\phi)+\mathrm{RBS}_{n,m}\sin(m\theta-nN_{\mathrm{fp}}\phi),
$$

$$
Z(\theta,\phi)=\sum_{m,n} \mathrm{ZBC}_{n,m}\cos(m\theta-nN_{\mathrm{fp}}\phi)+\mathrm{ZBS}_{n,m}\sin(m\theta-nN_{\mathrm{fp}}\phi),
$$

where:

- $N_{\mathrm{fp}}$ is the number of field periods (`NFP` in the VMEC input).

