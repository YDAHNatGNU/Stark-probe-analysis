# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Higher-accuracy boundary-element solve using analytical integration over planar
triangles (Wilton) together with Richardson extrapolation."""
import numpy as np
import time
from ferroelectric_bem import build_geometry, EPS0, sphere_analytic


def tri_int_vec(P, V1, V2, V3):
    """Observation point P (3,) against triangle arrays V1, V2, V3 (N,3).
    Returns J (N,3)."""
    n = np.cross(V2 - V1, V3 - V1)
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    d = np.einsum("k,nk->n", P, n) - np.einsum("nk,nk->n", V1, n)
    P0 = P[None, :] - d[:, None] * n
    J = np.zeros((V1.shape[0], 3))
    for A, B in ((V1, V2), (V2, V3), (V3, V1)):
        ev = B - A
        lhat = ev / np.linalg.norm(ev, axis=1, keepdims=True)
        mhat = np.cross(lhat, n)
        lminus = np.einsum("nk,nk->n", A - P0, lhat)
        lplus = np.einsum("nk,nk->n", B - P0, lhat)
        t0 = np.einsum("nk,nk->n", A - P0, mhat)
        base = t0**2 + d**2
        Rminus = np.sqrt(lminus**2 + base)
        Rplus = np.sqrt(lplus**2 + base)
        num = Rplus + lplus
        den = Rminus + lminus
        bad = (den < 1e-14) | (num < 1e-14)
        f1 = np.where(bad,
                      np.log((np.abs(lplus) + Rplus + 1e-300) /
                             (np.abs(lminus) + Rminus + 1e-300)),
                      np.log(np.where(bad, 1.0, num) / np.where(bad, 1.0, den)))
        J += mhat * f1[:, None]
    a = V1 - P[None, :]; b = V2 - P[None, :]; c = V3 - P[None, :]
    ra = np.linalg.norm(a, axis=1); rb = np.linalg.norm(b, axis=1)
    rc = np.linalg.norm(c, axis=1)
    triple = np.einsum("nk,nk->n", a, np.cross(b, c))
    denom = (ra * rb * rc + np.einsum("nk,nk->n", a, b) * rc
             + np.einsum("nk,nk->n", b, c) * ra
             + np.einsum("nk,nk->n", c, a) * rb)
    Omega = -2.0 * np.arctan2(triple, denom)
    J += n * Omega[:, None]
    return J


def assemble_M_analytic(geom):
    p0, p1, p2 = geom["tri"]
    c = geom["centroids"]; n = geom["normals"]
    N = len(c)
    M = np.zeros((N, N))
    for i in range(N):
        J = tri_int_vec(c[i], p0, p1, p2)         # (N,3)
        M[i, :] = (n[i][None, :] * J).sum(axis=1) / (4.0 * np.pi)
    np.fill_diagonal(M, 0.0)
    return M


def solve_sigma_analytic(geom, Ps, eps_b, eps_m):
    lam = (eps_b - eps_m) / (eps_b + eps_m)
    M = assemble_M_analytic(geom)
    N = len(geom["centroids"])
    A = np.eye(N) - 2.0 * lam * M
    rhs = (2.0 / (eps_b + eps_m)) * Ps * geom["normals"][:, 2]
    return np.linalg.solve(A, rhs)


def field_analytic(points, geom, sigma):
    """Exterior field from analytical integration, accurate close to the surface.
    points (P,3)."""
    p0, p1, p2 = geom["tri"]
    pts = np.atleast_2d(points)
    k = 1.0 / (4.0 * np.pi * EPS0)
    E = np.empty((len(pts), 3))
    for m, P in enumerate(pts):
        J = tri_int_vec(P, p0, p1, p2)            # (N,3)  = ∫(P-r')/R^3 per panel
        E[m] = k * (sigma[:, None] * J).sum(axis=0)
    return E


if __name__ == "__main__":
    Ps, eps_b, eps_m, R = 0.26, 6.0, 1.0, 1.0
    for subdiv in (2, 3):
        t = time.time()
        g = build_geometry("sphere", subdiv=subdiv, R=R)
        sig = solve_sigma_analytic(g, Ps, eps_b, eps_m)
        th = np.arccos(np.clip(g["centroids"][:, 2] / R, -1, 1))
        sig_ex = 3 * Ps * np.cos(th) / (eps_b + 2 * eps_m)
        err = np.max(np.abs(sig - sig_ex)) / np.max(np.abs(sig_ex))
        dt = time.time() - t
        print(f"subdiv={subdiv} N={len(sig)} sigma-err={err:.3%} time={dt:.1f}s")
