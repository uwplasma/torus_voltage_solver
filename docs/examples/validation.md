# Validation and convergence examples

These scripts are meant to build trust in the numerics by comparing against analytic expectations
and/or checking resolution convergence.

## Convergence sweep: max|Bn/B| for an axisymmetric toroidal field

Script:

- `examples/validation/convergence_bn_over_B_toroidal_current.py`

This is a “trust story” example: we prescribe a **net poloidal current** on the winding surface
using the current-potential model (a REGCOIL-style secular term), which generates an approximately
toroidal field inside the torus. For a purely toroidal field, the normalized normal component
should vanish on any interior torus:

$$
\frac{B_n}{|B|} = \frac{\mathbf B \cdot \hat{\mathbf n}}{|\mathbf B|} \approx 0.
$$

Run:

```bash
python examples/validation/convergence_bn_over_B_toroidal_current.py
```

The script sweeps winding-surface resolution and monitors:

- `max|Bn/B|` and `RMS(Bn/B)` on an interior target torus
- max relative error of $B_\phi(R)$ against Ampère’s-law scaling
  $B_\phi \approx \mu_0 I_\mathrm{pol}/(2\pi R)$

