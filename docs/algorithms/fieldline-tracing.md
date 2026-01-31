# Field-line tracing

`torus-solver` implements magnetic field-line tracing to visualize and validate the resulting fields.

## ODE for a field line

A magnetic field line satisfies:

$$
\frac{d\mathbf r}{ds} \parallel \mathbf B(\mathbf r).
$$

A common choice is to trace the normalized direction:

$$
\frac{d\mathbf r}{ds} = \mathbf b(\mathbf r) = \frac{\mathbf B(\mathbf r)}{|\mathbf B(\mathbf r)|},
$$

so that a constant step size corresponds roughly to an arc-length in meters.

## Numerical method

The code uses fixed-step RK4 and JAX control-flow:

- `jax.lax.scan` for the loop (fast and JIT-friendly)
- batch tracing via `trace_field_lines_batch` (scan over all lines simultaneously)

Implementation: `torus_solver/src/torus_solver/fieldline.py`.

## Interpreting drift from a target surface

Even if a surface has small $B_n/|B|$, field lines may still drift away depending on:

- how the target surface sits relative to the true invariant surfaces of the field
- numerical integration step size
- whether the objective enforced tangency only at discrete points

In `examples/optimize_vmec_surface_Bn.py`, a simple drift diagnostic measures the distance of traced points to a discrete point cloud of the target surface.

:::{tip}
When diagnosing drift, always check convergence with smaller step sizes and compare multiple seeds.
:::

