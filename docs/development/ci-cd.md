# CI/CD and Read the Docs

This project is designed to be:

- tested on every pull request
- able to build documentation in CI and on Read the Docs
- publishable to PyPI via GitHub Actions

## GitHub Actions

Two workflows are provided:

- **CI**: runs unit tests, checks that code compiles, builds the docs, and builds the Python package artifacts.
- **Publish**: on tagged releases, builds and uploads distributions to PyPI.

See `.github/workflows/`.

## Read the Docs

Read the Docs is configured via `.readthedocs.yaml` to:

1. Install the package with the `docs` extra.
2. Build documentation using `docs/conf.py`.

If you change doc dependencies, update:

- `pyproject.toml` (`[project.optional-dependencies].docs`)
- `.readthedocs.yaml`

## PyPI publishing

The publish workflow is written to support PyPI “trusted publishing” (OIDC).

To enable uploads:

1. Create a project on PyPI with the same name as `pyproject.toml` (`torus-solver`).
2. Configure the repository as a trusted publisher on PyPI.
3. Tag a release, e.g. `v0.1.0`.

:::{note}
If you prefer API tokens instead of OIDC, you can adapt the workflow to use `TWINE_PASSWORD` / `PYPI_API_TOKEN` secrets. OIDC is recommended for security.
:::

