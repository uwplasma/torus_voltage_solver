# Geometry: the circular torus surface

`torus-solver` uses a circular torus surface as the winding surface.

## Parameterization

We use:

- $\theta \in [0, 2\pi)$: poloidal angle (around the small circle)
- $\phi \in [0, 2\pi)$: toroidal angle (around the large circle)

with:

- major radius $R_0$
- minor radius $a$
- $R(\theta) = R_0 + a\cos\theta$

The surface embedding is

$$
\mathbf r(\theta,\phi) =
\begin{pmatrix}
R(\theta)\cos\phi\\
R(\theta)\sin\phi\\
a\sin\theta
\end{pmatrix}.
$$

## Tangent vectors and metric

The tangent vectors are

$$
\mathbf r_\theta = \frac{\partial \mathbf r}{\partial \theta},\qquad
\mathbf r_\phi = \frac{\partial \mathbf r}{\partial \phi}.
$$

For the circular torus, the metric coefficients are:

$$
E = \mathbf r_\theta \cdot \mathbf r_\theta = a^2,\qquad
F = \mathbf r_\theta \cdot \mathbf r_\phi = 0,\qquad
G = \mathbf r_\phi \cdot \mathbf r_\phi = R(\theta)^2.
$$

The surface area element is:

$$
dA = \sqrt{g}\, d\theta d\phi,\qquad
\sqrt{g} = |\mathbf r_\theta \times \mathbf r_\phi| = a R(\theta).
$$

## Unit normal

We define the (outward) unit normal

$$
\hat{\mathbf n} = \frac{\mathbf r_\theta \times \mathbf r_\phi}{|\mathbf r_\theta \times \mathbf r_\phi|}.
$$

The sign of $\hat{\mathbf n}$ matters for conventions (e.g. the sign of $B\cdot \hat{\mathbf n}$), but the common objective uses $|B\cdot\hat{\mathbf n}|/|B|$, which is sign-invariant.

## Where this lives in the code

- Surface construction: `src/torus_solver/torus.py`
  - `make_torus_surface(...)` produces:
    - `r`, `r_theta`, `r_phi`
    - metric/area quantities like `sqrt_g`, `G`, `area_weights`
