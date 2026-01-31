# Biot–Savart: magnetic field from surface currents

Given a surface current density $\mathbf K(\mathbf r')$ on the winding surface $S$, the magnetic field at a point $\mathbf x$ is:

$$
\mathbf B(\mathbf x) =
\frac{\mu_0}{4\pi}
\iint_{S}
\frac{
\mathbf K(\mathbf r') \times (\mathbf x - \mathbf r')
}{
|\mathbf x - \mathbf r'|^3
}\, dA'.
$$

## Discretization

On a uniform $(\theta,\phi)$ grid, the surface integral becomes a weighted sum:

$$
\mathbf B(\mathbf x) \approx
\frac{\mu_0}{4\pi}
\sum_{j,k}
\frac{
\mathbf K_{jk} \times (\mathbf x - \mathbf r_{jk})
}{
|\mathbf x - \mathbf r_{jk}|^3
}\, \Delta A_{jk},
$$

where:

- $\mathbf r_{jk} = \mathbf r(\theta_j,\phi_k)$
- $\Delta A_{jk} = \sqrt{g(\theta_j)}\, \Delta\theta\, \Delta\phi$

## Numerical stability: softening

If $\mathbf x$ is extremely close to the surface, the kernel becomes singular. In this prototype we use a small softening length $\varepsilon$:

$$
|\mathbf x-\mathbf r'|^2 \to |\mathbf x-\mathbf r'|^2 + \varepsilon^2,
$$

which is controlled by `biot_savart_eps` in scripts.

## Performance considerations

The direct sum is $O(N_\text{eval} N_\text{surf})$. For research workflows this is often acceptable at moderate resolutions, but larger problems require acceleration strategies (future work), such as:

- fast multipole methods (FMM)
- FFT-based convolution methods (with careful treatment of geometry)
- hierarchical / adaptive sampling

In this repository, `src/torus_solver/biot_savart.py` implements a **chunked** evaluation mode to reduce peak memory pressure when `N_eval` is large.
