# torus-solver

JAX-differentiable surface-current modeling and optimization on a circular torus, with targets motivated by stellarators and tokamaks.

This documentation is written for two audiences:

1. **Newcomers**: you can start from scratch, learn the physics assumptions and equations, and run the examples end-to-end.
2. **Domain experts**: you can quickly map the equations to the implementation details, understand algorithmic tradeoffs, and extend the code.

## Gallery

```{figure} _static/images/helical_fieldlines_final.png
:alt: Electrode-driven helical optimization: final field lines
:width: 95%

Electrode-driven helical optimization: 3D field lines from the Biot–Savart field of the optimized surface currents.
```

```{figure} _static/images/vmec_Bn_over_B_best.png
:alt: VMEC target surface: optimized Bn/B map
:width: 95%

VMEC target surface: optimized normalized normal field $B_n/|B|$ (REGCOIL-like current-potential model).
```

```{figure} _static/images/vmec_regK_scan_lcurve.png
:alt: Regularization scan tradeoff curve
:width: 95%

Regularization scan (“L-curve”-style): tradeoff between current magnitude ($K_{\mathrm{rms}}$) and field quality (max$|B_n/B|$).
```

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
