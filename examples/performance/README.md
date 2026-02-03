# Performance

These scripts highlight performance/scaling and “why JAX?”.

- `jax_vs_numpy_biot_savart_speed.py`: timing comparison between a simple NumPy baseline and JAX (`jax.jit`) Biot–Savart evaluation.

Run:

```bash
python examples/performance/jax_vs_numpy_biot_savart_speed.py --n-eval 512 --repeat 20
```

