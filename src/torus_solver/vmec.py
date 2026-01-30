from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class VMECBoundary:
    """Fourier representation of a VMEC boundary surface.

    VMEC convention (stellarator-symmetric case):
      R(θ,φ) = Σ [RBC(n,m) cos(mθ - n*nfp*φ) + RBS(n,m) sin(mθ - n*nfp*φ)]
      Z(θ,φ) = Σ [ZBC(n,m) cos(mθ - n*nfp*φ) + ZBS(n,m) sin(mθ - n*nfp*φ)]

    Here we use φ on [0,2π) for the full torus, so the nfp factor is included.
    """

    nfp: int
    m: np.ndarray  # (Nmodes,)
    n: np.ndarray  # (Nmodes,)
    rbc: np.ndarray  # (Nmodes,)
    rbs: np.ndarray  # (Nmodes,)
    zbc: np.ndarray  # (Nmodes,)
    zbs: np.ndarray  # (Nmodes,)


_NFP_RE = re.compile(r"(?im)^[ \t]*NFP[ \t]*=[ \t]*([+-]?\d+)")
_COEF_RE = re.compile(
    r"(?P<name>RBC|RBS|ZBC|ZBS)\(\s*(?P<n>[+-]?\d+)\s*,\s*(?P<m>[+-]?\d+)\s*\)\s*=\s*(?P<val>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?)"
)


def read_vmec_boundary(path: str | Path) -> VMECBoundary:
    """Parse a VMEC input file and return the boundary Fourier coefficients."""
    path = Path(path)
    text = path.read_text()

    m_nfp = _NFP_RE.search(text)
    if m_nfp is None:
        raise ValueError(f"Could not find NFP in VMEC input: {path}")
    nfp = int(m_nfp.group(1))

    coeffs: dict[str, dict[tuple[int, int], float]] = {k: {} for k in ("RBC", "RBS", "ZBC", "ZBS")}
    for m in _COEF_RE.finditer(text):
        name = m.group("name")
        n = int(m.group("n"))
        mm = int(m.group("m"))
        val = float(m.group("val").replace("D", "E").replace("d", "e"))
        coeffs[name][(n, mm)] = val

    keys = set().union(*[set(d.keys()) for d in coeffs.values()])
    if not keys:
        raise ValueError(f"No boundary coefficients (RBC/RBS/ZBC/ZBS) found in: {path}")

    # Sort for determinism.
    keys_sorted = sorted(keys, key=lambda nm: (nm[1], nm[0]))  # (m, n)
    n_arr = np.array([n for (n, m) in keys_sorted], dtype=int)
    m_arr = np.array([m for (n, m) in keys_sorted], dtype=int)

    def get(name: str) -> np.ndarray:
        return np.array([coeffs[name].get((n, m), 0.0) for (n, m) in keys_sorted], dtype=float)

    return VMECBoundary(
        nfp=nfp,
        m=m_arr,
        n=n_arr,
        rbc=get("RBC"),
        rbs=get("RBS"),
        zbc=get("ZBC"),
        zbs=get("ZBS"),
    )


def vmec_boundary_RZ_and_derivatives(
    boundary: VMECBoundary, *, theta: jnp.ndarray, phi: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Evaluate (R,Z) and first derivatives on a (theta,phi) grid.

    theta: (Nθ,), phi: (Nφ,) with phi spanning [0,2π) for the full torus.
    Returns arrays of shape (Nθ,Nφ): (R,Z,R_theta,R_phi,Z_theta,Z_phi).
    """
    theta = jnp.asarray(theta, dtype=jnp.float64)[:, None]  # (Nθ,1)
    phi = jnp.asarray(phi, dtype=jnp.float64)[None, :]  # (1,Nφ)

    m = jnp.asarray(boundary.m, dtype=jnp.float64)[:, None, None]  # (M,1,1)
    n = jnp.asarray(boundary.n, dtype=jnp.float64)[:, None, None]  # (M,1,1)
    nfp = float(boundary.nfp)

    ang = m * theta[None, :, :] - (n * nfp) * phi[None, :, :]
    c = jnp.cos(ang)
    s = jnp.sin(ang)

    rbc = jnp.asarray(boundary.rbc, dtype=jnp.float64)[:, None, None]
    rbs = jnp.asarray(boundary.rbs, dtype=jnp.float64)[:, None, None]
    zbc = jnp.asarray(boundary.zbc, dtype=jnp.float64)[:, None, None]
    zbs = jnp.asarray(boundary.zbs, dtype=jnp.float64)[:, None, None]

    R = jnp.sum(rbc * c + rbs * s, axis=0)
    Z = jnp.sum(zbc * c + zbs * s, axis=0)

    # Derivatives with respect to theta:
    # d/dθ cos(mθ-...) = -m sin(...)
    # d/dθ sin(mθ-...) =  m cos(...)
    R_theta = jnp.sum((-m) * rbc * s + (m) * rbs * c, axis=0)
    Z_theta = jnp.sum((-m) * zbc * s + (m) * zbs * c, axis=0)

    # Derivatives with respect to phi (full-torus phi so includes nfp):
    # d/dφ cos(mθ-n*nfp*φ) = + (n*nfp) sin(...)
    # d/dφ sin(mθ-n*nfp*φ) = - (n*nfp) cos(...)
    nn = n * nfp
    R_phi = jnp.sum((nn) * rbc * s + (-nn) * rbs * c, axis=0)
    Z_phi = jnp.sum((nn) * zbc * s + (-nn) * zbs * c, axis=0)

    return R, Z, R_theta, R_phi, Z_theta, Z_phi


def vmec_boundary_xyz_and_normals(
    boundary: VMECBoundary, *, theta: jnp.ndarray, phi: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return xyz and outward unit normal on the VMEC boundary."""
    R, Z, R_theta, R_phi, Z_theta, Z_phi = vmec_boundary_RZ_and_derivatives(boundary, theta=theta, phi=phi)

    phi2 = jnp.asarray(phi, dtype=jnp.float64)[None, :]
    c = jnp.cos(phi2)
    s = jnp.sin(phi2)

    x = R * c
    y = R * s
    z = Z
    xyz = jnp.stack([x, y, z], axis=-1)

    # Tangents.
    r_theta = jnp.stack([R_theta * c, R_theta * s, Z_theta], axis=-1)
    r_phi = jnp.stack([R_phi * c - R * s, R_phi * s + R * c, Z_phi], axis=-1)

    n = jnp.cross(r_theta, r_phi)
    n_hat = n / (jnp.linalg.norm(n, axis=-1, keepdims=True) + 1e-30)
    return xyz, n_hat

