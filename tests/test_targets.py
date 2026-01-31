import tempfile
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from torus_solver.targets import vmec_target_surface
from torus_solver.vmec import read_vmec_boundary


def test_vmec_target_surface_fits_inside_torus_margin():
    # Axisymmetric circular cross section:
    #   R = R0 + a cos(theta), Z = a sin(theta)
    R0 = 1.0
    a = 0.3
    txt = f"""&INDATA
NFP = 1
RBC(   0,   0) =  {R0:.16e},    ZBS(   0,   0) =  0.0
RBC(   0,   1) =  {a:.16e},    ZBS(   0,   1) =  {a:.16e}
/"""

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "input.test"
        path.write_text(txt)
        boundary = read_vmec_boundary(path)

    fit_margin = 0.7
    target = vmec_target_surface(
        boundary,
        torus_R0=R0,
        torus_a=a,
        fit_margin=fit_margin,
        n_theta=32,
        n_phi=40,
        dtype=jnp.float64,
    )

    xyz = np.asarray(target.xyz)
    R = np.sqrt(xyz[..., 0] ** 2 + xyz[..., 1] ** 2)
    Z = xyz[..., 2]
    rho = np.sqrt((R - R0) ** 2 + Z**2)
    assert float(np.max(rho)) <= fit_margin * a + 1e-12

