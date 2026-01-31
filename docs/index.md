# torus-solver

JAX-differentiable surface-current modeling and optimization on a circular torus, with targets motivated by stellarators and tokamaks.

This documentation is written for two audiences:

1. **Newcomers**: you can start from scratch, learn the physics assumptions and equations, and run the examples end-to-end.
2. **Domain experts**: you can quickly map the equations to the implementation details, understand algorithmic tradeoffs, and extend the code.

```{toctree}
:maxdepth: 2
:caption: Getting Started

getting-started
installation
examples/index
applications
glossary
```

```{toctree}
:maxdepth: 2
:caption: Theory

theory/geometry
theory/surface-operators
theory/electrode-model
theory/current-potential
theory/biot-savart
```

```{toctree}
:maxdepth: 2
:caption: Algorithms

algorithms/optimization
algorithms/fieldline-tracing
algorithms/validation
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api/index
```

```{toctree}
:maxdepth: 2
:caption: Development

development/contributing
development/ci-cd
development/roadmap
references
```
