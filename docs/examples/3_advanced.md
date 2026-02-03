# 3. Advanced examples

These scripts are closer to “research workflows”:

- fitting a VMEC boundary surface inside a circular torus winding surface
- minimizing the **normalized normal field** $B_n/|B|$ on that target surface
- scanning regularization (REGCOIL-style “L-curve” tradeoffs)
- more complicated geometry targets (toy near-axis constructions)

The default VMEC input shipped with the repo lives at:

- `examples/data/vmec/input.QA_nfp2`

## VMEC target: minimize normalized $B_n$

Script:

- `examples/3_advanced/optimize_vmec_surface_Bn.py`

The main objective used here is:

$$
\mathrm{rms}(B_n/|B|) = \sqrt{\langle (\mathbf B\cdot\hat{\mathbf n}/|\mathbf B|)^2 \rangle},
$$

evaluated on the target (VMEC) surface, optionally with a weighted $p$-norm to penalize peaks.

Two models are supported:

1. **Current potential** (`--model current-potential`, recommended): a REGCOIL-like single-valued current potential plus optional net poloidal/toroidal currents.
2. **Electrodes** (`--model electrodes`): discrete current injection/extraction on the winding surface, plus an imposed analytic background toroidal field.

### Current-potential model (REGCOIL-like)

This model represents the winding-surface current as:

$$
\mathbf K = \hat{\mathbf n}\times\nabla_s \Phi,
$$

with optional secular terms that set net currents:

$$
\Phi(\theta,\phi) =
\Phi_{\mathrm{sv}}(\theta,\phi)
 + \frac{I_{\mathrm{tor}}}{2\pi}\,\theta
 + \frac{I_{\mathrm{pol}}}{2\pi}\,\phi.
$$

The example parameterizes $\Phi_{\mathrm{sv}}$ as a truncated Fourier series compatible with an
$N_{\mathrm{fp}}$-periodic VMEC boundary.

The objective uses a weighted $p$-norm proxy to penalize peaks in the normalized normal field
$B_n/\|\mathbf B\|$:

$$
\left\|B_n/\|\mathbf B\|\right\|_{p,w}
  = \left(\frac{\sum_i w_i\,\left| (B_n/\|\mathbf B\|)_i \right|^p}{\sum_i w_i}\right)^{1/p}.
$$

Run the REGCOIL-like model:

```bash
python examples/3_advanced/optimize_vmec_surface_Bn.py \
  --model current-potential \
  --vmec-input examples/data/vmec/input.QA_nfp2
```

Outputs include:

- target maps of $B_n/|B|$ (init/final)
- winding surface current maps
- field-line tracing + a simple drift diagnostic
- `figures/optimize_vmec_surface_Bn/paraview/scene.vtm`

## Regularization scan (tradeoff curve)

Script:

- `examples/3_advanced/scan_vmec_surface_regularization.py`

This sweeps the current-magnitude regularization weight and produces a tradeoff curve between:

- field quality (e.g. $\max\left|\frac{B_n}{\|\mathbf B\|}\right|$ on the target surface), and
- current magnitude ($K_{\mathrm{rms}}$ on the winding surface).

```bash
python examples/3_advanced/scan_vmec_surface_regularization.py \
  --vmec-input examples/data/vmec/input.QA_nfp2
```

Outputs include:

- `l_curve.png`
- `Bn_over_B_best.png`, `Kmag_best.png`
- `figures/scan_vmec_surface_regularization/paraview/scene.vtm`

## Toy near-axis target: bumpy axis + rotating ellipse

Script:

- `examples/3_advanced/optimize_bumpy_axis_rotating_ellipse.py`

This is a deliberately simplified construction inspired by near-axis intuition:

- prescribe a bumpy axis $R(\phi), Z(\phi)$
- build a discrete Frenet-like frame along the axis
- define evaluation points on a rotating ellipse around the axis
- optimize electrodes so the on-ellipse field matches a prescribed target

```bash
python examples/3_advanced/optimize_bumpy_axis_rotating_ellipse.py --n-steps 120
```

Optional background field (for *field-line tracing only*):

- `--Bext0 <Tesla>` adds an external ideal toroidal field $B_\phi = B_{\mathrm{ext0}} (R_0/R)$ to the traced field lines.
- `--Bpol0 <Tesla>` adds an external tokamak-like poloidal component
  $B_\theta = B_{\mathrm{pol0}} (R_0/R)$ to the traced field lines.

Use `--Bext0 0 --Bpol0 0` to disable.

Outputs include:

- target-geometry diagnostics
- 3D field lines
- `figures/optimize_bumpy_axis_rotating_ellipse/paraview/scene.vtm`

## Interactive VMEC optimization GUI (VTK)

Script:

- `examples/3_advanced/gui_optimize_vmec_surface_Bn.py`

```bash
pip install -e '.[gui]'
python examples/3_advanced/gui_optimize_vmec_surface_Bn.py --vmec-input examples/data/vmec/input.QA_nfp2
```

Press `E` to export the current state to ParaView.

Field-line background toggle:

- press `T` to toggle including the background toroidal $B_\phi\propto 1/R$ field in **field-line tracing** (visualization only).
- press `Y` to toggle including the background poloidal $B_\theta\propto 1/R$ field in **field-line tracing** (visualization only).
- press `[`/`]` to decrease/increase `B0`, and `,`/`.` to decrease/increase `Bpol0`.

The optimization objective and the displayed target-surface `(B·n)/|B|` always use the configured `B0` and `Bpol0`.
