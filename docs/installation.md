# Installation

## Requirements

- Python **3.10+**
- `jax` + `jaxlib` (CPU works out of the box for most platforms)
- Standard scientific Python: NumPy, Matplotlib

## Install (editable, recommended for research)

From the repo root (`torus_voltage_solver/`):

```bash
pip install -e .
```

If your environment has restricted network access during build isolation:

```bash
pip install -e . --no-build-isolation
```

## Optional extras

- GUI (VTK):

```bash
pip install -e '.[gui]'
```

- Docs (Sphinx / Read the Docs):

```bash
pip install -e '.[docs]'
```

- Tests:

```bash
pip install -e '.[test]'
pytest -q
```

## Build the documentation locally

From the repo root (`torus_voltage_solver/`):

```bash
pip install -e '.[docs]'
python -m sphinx -b html docs docs/_build/html
open docs/_build/html/index.html
```

:::{note}
The documentation uses MathJax for equations. This works in the built HTML output automatically.
:::
