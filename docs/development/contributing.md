# Contributing

This repository is intended to be a research codebase that remains readable and extensible.

## Development install

From `torus_solver/`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[test,docs]'
```

## Run tests

```bash
pytest -q
```

## Build docs locally

```bash
python -m sphinx -b html docs docs/_build/html -W
```

## Benchmarks

Benchmarks live in `benchmarks/` and are meant to be fast sanity checks rather than exhaustive performance reports:

```bash
python benchmarks/bench_forward_and_grad.py
python benchmarks/bench_fieldline.py
```

## Code organization and extension points

If you want to add new research features, common extension points are:

- New winding-surface geometry (replace/extend `TorusSurface`)
- New current representations (new parameterization of $\Phi$ or a different electrode kernel)
- Faster Biot–Savart backends (FMM, batching, caching, GPU-friendly kernels)
- New objective terms and constraints (e.g. coil power, smoothness, symmetry, manufacturing rules)
- Better field-line diagnostics (surface-of-section, rotational transform, island detection)

## Scientific contribution guidelines

- Include **units** and sign conventions in new code and docs.
- Add at least one **validation test** for any new physical model component.
- When changing an algorithm, add a small regression test that would have caught common mistakes (e.g. wrong factor of $2\pi$, wrong normalization, wrong current sign).

