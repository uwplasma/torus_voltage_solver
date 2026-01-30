import tempfile
from pathlib import Path

import numpy as np

from torus_solver.vmec import read_vmec_boundary, vmec_boundary_RZ_and_derivatives


def test_vmec_boundary_parser_and_nfp_factor():
    # Minimal VMEC-like input with one toroidal cosine mode.
    txt = """&INDATA
NFP = 2
RBC(   0,   0) =  1.0,    ZBS(   0,   0) =  0.0
RBC(   1,   0) =  0.1,    ZBS(   1,   0) =  0.0
/"""

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "input.test"
        path.write_text(txt)
        b = read_vmec_boundary(path)

    assert b.nfp == 2

    theta = np.array([0.0, 1.0])
    phi = np.array([0.0, np.pi / 2])  # cos(2*phi) toggles between +1 and -1

    import jax.numpy as jnp

    R, Z, *_ = vmec_boundary_RZ_and_derivatives(b, theta=jnp.asarray(theta), phi=jnp.asarray(phi))

    R_np = np.asarray(R)
    Z_np = np.asarray(Z)

    # R = 1 + 0.1*cos(2*phi). At phi=0 -> 1.1, at phi=pi/2 -> 0.9.
    np.testing.assert_allclose(R_np[:, 0], 1.1, rtol=0, atol=1e-12)
    np.testing.assert_allclose(R_np[:, 1], 0.9, rtol=0, atol=1e-12)
    np.testing.assert_allclose(Z_np, 0.0, rtol=0, atol=1e-12)

