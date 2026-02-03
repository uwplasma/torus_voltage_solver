# Performance examples

These scripts highlight performance/scaling and “why JAX?”.

## JAX vs NumPy: Biot–Savart timing

Script:

- `examples/performance/jax_vs_numpy_biot_savart_speed.py`

This compares:

- a simple NumPy baseline that loops over evaluation points, and
- the vectorized JAX implementation (compiled with `jax.jit`).

Run:

```bash
python examples/performance/jax_vs_numpy_biot_savart_speed.py --n-eval 512 --repeat 20
```

