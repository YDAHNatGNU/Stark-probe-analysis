# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Three-region core-shell boundary-element solver.

Regions: core (eps_b, spontaneous polarisation Ps) | shell (eps_s) | medium (eps_m).
An equivalent surface charge is placed on both interfaces and represented as a
single layer in vacuum, so that continuity of the potential is automatic. Reduces
exactly to the single-interface expression in the limit of no shell. Includes the
concentric-sphere analytical solution used for validation."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
from ferroelectric_bem import build_geometry, EPS0
from accurate_bem import tri_int_vec


# --------------------------------------------------------------
def offset_mesh(geom, t):
    """Build a mesh offset by t along the vertex normals (uniform shell thickness)."""
    v = geom["verts"].copy()
    f = geom["faces"]
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)
    area2 = np.linalg.norm(fn, axis=1, keepdims=True)
    fn_unit = fn / area2
    cen = (p0 + p1 + p2) / 3.0
    flip = np.sum(fn_unit * cen, axis=1) < 0
    fn_unit[flip] *= -1
    vn = np.zeros_like(v)
    for k in range(3):
        np.add.at(vn, f[:, k], fn_unit * area2)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True)
    v_out = v + t * vn
    return _mesh_from(v_out, f)


def _mesh_from(v, f):
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    cen = (p0 + p1 + p2) / 3.0
    cr = np.cross(p1 - p0, p2 - p0)
    ar = 0.5 * np.linalg.norm(cr, axis=1)
    nrm = cr / (2.0 * ar[:, None])
    flip = np.sum(nrm * cen, axis=1) < 0
    nrm[flip] *= -1
    return dict(verts=v, faces=f, tri=(p0, p1, p2), centroids=cen,
                areas=ar, normals=nrm)


# --------------------------------------------------------------
def assemble_M_multi(geoms):
    """Full double-layer matrix for all surfaces combined (analytical integration)."""
    P0 = np.vstack([g["tri"][0] for g in geoms])
    P1 = np.vstack([g["tri"][1] for g in geoms])
    P2 = np.vstack([g["tri"][2] for g in geoms])
    C = np.vstack([g["centroids"] for g in geoms])
    N = np.vstack([g["normals"] for g in geoms])
    n = len(C)
    M = np.zeros((n, n))
    for i in range(n):
        J = tri_int_vec(C[i], P0, P1, P2)
        M[i, :] = (N[i][None, :] * J).sum(axis=1) / (4.0 * np.pi)
    np.fill_diagonal(M, 0.0)
    return M, C, N, np.concatenate([g["areas"] for g in geoms])


def solve_core_shell(g_core, g_shell, Ps, eps_b, eps_s, eps_m):
    """Returns sigma for all surfaces, the index split points, and the merged geometry.
    M is converted to A in place and solved with overwrite, to save memory."""
    from scipy.linalg import solve as lasolve
    M, C, Nrm, Ar = assemble_M_multi([g_core, g_shell])
    n1 = len(g_core["centroids"])
    n = len(C)
    lam = np.empty(n)
    lam[:n1] = (eps_s - eps_b) / (eps_s + eps_b)      # core/shell
    lam[n1:] = (eps_m - eps_s) / (eps_m + eps_s)      # shell/medium
    b = np.zeros(n)
    b[:n1] = (2.0 / (eps_s + eps_b)) * Ps * Nrm[:n1, 2]
    M *= (2.0 * lam)[:, None]
    M[np.diag_indices(n)] += 1.0                       # + I
    sigma = lasolve(M, b, overwrite_a=True, overwrite_b=False,
                    check_finite=False)
    del M
    return sigma, n1, C, Nrm, Ar, (g_core, g_shell)


def field_core_shell(points, geoms, sigma):
    """Exterior field from analytical integration."""
    P0 = np.vstack([g["tri"][0] for g in geoms])
    P1 = np.vstack([g["tri"][1] for g in geoms])
    P2 = np.vstack([g["tri"][2] for g in geoms])
    pts = np.atleast_2d(points)
    k = 1.0 / (4.0 * np.pi * EPS0)
    E = np.empty((len(pts), 3))
    for m, Pp in enumerate(pts):
        J = tri_int_vec(Pp, P0, P1, P2)
        E[m] = k * (sigma[:, None] * J).sum(axis=0)
    return E


# --------------------------------------------------------------
def analytic_D(Ps, eps_b, eps_s, eps_m, R1, R2):
    """Exterior dipole coefficient D of the concentric-sphere solution
    (phi_out = D cos(theta)/r^2)."""
    num = -3.0 * Ps * R1**3 * R2**3 * eps_s
    den = EPS0 * (2*R1**3*(eps_b*eps_m - eps_b*eps_s - eps_m*eps_s + eps_s**2)
                  - R2**3*(2*eps_b*eps_m + eps_b*eps_s
                           + 4*eps_m*eps_s + 2*eps_s**2))
    return num / den


def analytic_E(points, Ps, eps_b, eps_s, eps_m, R1, R2):
    D = analytic_D(Ps, eps_b, eps_s, eps_m, R1, R2)
    pts = np.atleast_2d(points)
    r = np.linalg.norm(pts, axis=1)
    cth = pts[:, 2] / r
    return np.abs(D) / r**3 * np.sqrt(1 + 3*cth**2)


# --------------------------------------------------------------
if __name__ == "__main__":
    Ps, eps_b, eps_s, eps_m = 0.26, 6.0, 3.9, 24.0
    R1, t = 1.0, 0.20
    R2 = R1 + t
    print("=== Core-shell sphere validation (SiO2 shell) ===")
    print(f"  R1={R1}, t={t}, eps_b={eps_b}, eps_s={eps_s}, eps_m={eps_m}")
    for sub in (2, 3):
        gc = build_geometry("sphere", subdiv=sub, R=R1)
        gs = build_geometry("sphere", subdiv=sub, R=R2)
        sig, n1, C, Nrm, Ar, gg = solve_core_shell(gc, gs, Ps,
                                                   eps_b, eps_s, eps_m)
        th = np.linspace(0.05, np.pi-0.05, 25)
        errs = []
        for rf in (1.2, 1.5, 2.0, 3.0):
            r = rf * R2
            pts = np.column_stack([r*np.sin(th), np.zeros_like(th),
                                   r*np.cos(th)])
            Eb = np.linalg.norm(field_core_shell(pts, gg, sig), axis=1)
            Ea = analytic_E(pts, Ps, eps_b, eps_s, eps_m, R1, R2)
            errs.append(np.max(np.abs(Eb-Ea)/Ea))
        print(f"  subdiv={sub} (N={len(sig)}): max error "
              + "  ".join(f"r={rf}R2:{e:.2%}"
                          for rf, e in zip((1.2,1.5,2.0,3.0), errs)))
    print("\n=== No-shell limit (eps_s = eps_m) ===")
    from ferroelectric_bem import sphere_analytic
    gc = build_geometry("sphere", subdiv=3, R=R1)
    gs = build_geometry("sphere", subdiv=3, R=R2)
    sig, n1, C, Nrm, Ar, gg = solve_core_shell(gc, gs, Ps, eps_b,
                                               eps_m, eps_m)
    pts = np.array([[0, 0, 2.0]])
    Eb = np.linalg.norm(field_core_shell(pts, gg, sig))
    _, Ea, _, _ = sphere_analytic(pts, Ps, eps_b, eps_m, R1)
    print(f"  BEM={Eb:.4e}  single-interface analytical={Ea[0]:.4e}  "
          f"error={abs(Eb-Ea[0])/Ea[0]:.2%}")
