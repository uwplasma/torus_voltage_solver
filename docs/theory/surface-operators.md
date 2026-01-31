# Surface differential operators on the torus

Two models in this repository require surface differential operators:

1. The **electrode model** solves a Poisson equation on the surface (Laplace–Beltrami operator).
2. The **current-potential model** computes $\nabla_s \Phi$ to obtain a divergence-free surface current.

## Surface gradient

For a scalar field $f(\theta,\phi)$ on a parameterized surface, the surface gradient can be written as:

$$
\nabla_s f = g^{ij} \frac{\partial f}{\partial u^i}\, \mathbf r_{u^j},
$$

where $u^1=\theta$, $u^2=\phi$, and $g^{ij}$ is the inverse metric.

For the circular torus (orthogonal coordinates, $F=0$):

$$
\nabla_s f = \frac{1}{E}\, f_\theta\, \mathbf r_\theta + \frac{1}{G}\, f_\phi\, \mathbf r_\phi
= \frac{1}{a^2}\, f_\theta\, \mathbf r_\theta + \frac{1}{R(\theta)^2}\, f_\phi\, \mathbf r_\phi.
$$

## Laplace–Beltrami (surface Laplacian)

The Laplace–Beltrami operator is:

$$
\Delta_s f = \nabla_s \cdot \nabla_s f
=
\frac{1}{\sqrt{g}}
\left[
\frac{\partial}{\partial \theta}
\left(\sqrt{g}\, g^{\theta\theta}\, f_\theta\right)
+
\frac{\partial}{\partial \phi}
\left(\sqrt{g}\, g^{\phi\phi}\, f_\phi\right)
\right].
$$

For the circular torus, with $\sqrt{g}=aR(\theta)$, $g^{\theta\theta}=1/a^2$, $g^{\phi\phi}=1/R(\theta)^2$, this becomes:

$$
\Delta_s f
=
\frac{1}{a^2 R(\theta)} \frac{\partial}{\partial \theta}\left(R(\theta)\, f_\theta\right)
+
\frac{1}{R(\theta)^2}\, f_{\phi\phi}.
$$

## Discretization and numerics in this code

### Periodic grids

We discretize $\theta$ and $\phi$ on uniform periodic grids:

$$
\theta_j = \frac{2\pi j}{N_\theta},\qquad
\phi_k = \frac{2\pi k}{N_\phi}.
$$

### Spectral derivatives

Derivatives in $\theta$ and $\phi$ are computed using FFTs:

$$
\frac{\partial f}{\partial x} \approx \mathcal{F}^{-1}\left( i k\, \mathcal{F}(f)\right),
$$

implemented in `src/torus_solver/spectral.py`.

### Poisson solve

Because $R(\theta)$ varies, $\Delta_s$ is **not diagonal** in Fourier space, so the Poisson equation is solved with Conjugate Gradient (CG) using a matrix-free operator.

In the electrode model we solve (schematically):

$$
-\Delta_s V = s/\sigma_s,
$$

with:

- a **net-zero** constraint on $s$ (closed surface)
- a **gauge** fix on $V$ (remove the constant nullspace)

See `src/torus_solver/poisson.py`.
