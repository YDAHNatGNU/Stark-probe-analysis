# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Boundary-element solution for the field of a single-domain ferroelectric
nanoparticle. Formulation: frozen spontaneous polarisation with linear dielectric
response, solved as a Fredholm integral equation of the second kind for the bound
surface charge. Provides the icosphere mesh, the superellipsoid map, assembly of
the geometric matrix M, the single-interface solve and the analytical sphere
reference used for validation.

Note: eps_b is the background (non-soft-mode) permittivity of BaTiO3, not the
static value, so that the spontaneous polarisation is not counted twice."""
import numpy as np

EPS0 = 8.8541878128e-12  # F/m


# --------------------------------------------------------------------
# --------------------------------------------------------------------
def icosphere(subdiv=3):
    """Unit-sphere triangular mesh by icosahedron subdivision.
    Returns verts (V,3) and faces (F,3)."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=float)
    verts /= np.linalg.norm(verts, axis=1, keepdims=True)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=int)

    for _ in range(subdiv):
        mid = {}
        new_faces = []
        vlist = verts.tolist()

        def midpoint(a, b):
            key = (min(a, b), max(a, b))
            if key in mid:
                return mid[key]
            m = (np.array(vlist[a]) + np.array(vlist[b])) / 2.0
            m /= np.linalg.norm(m)
            vlist.append(m.tolist())
            idx = len(vlist) - 1
            mid[key] = idx
            return idx

        for (a, b, c) in faces:
            ab = midpoint(a, b)
            bc = midpoint(b, c)
            ca = midpoint(c, a)
            new_faces += [[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]
        verts = np.array(vlist)
        faces = np.array(new_faces, dtype=int)

    return verts, faces


def superellipsoid_map(dirs, a, p):
    """Map unit direction vectors onto the superellipsoid
    |x/a|^p + |y/a|^p + |z/a|^p = 1. p = 2 gives a sphere; larger p approaches a
    sharp cube (rounded cube: p ~ 4-6).
    """
    u = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    denom = np.sum(np.abs(u / a) ** p, axis=1) ** (1.0 / p)
    return u / denom[:, None]


def build_geometry(kind="sphere", subdiv=3, R=1.0, a=1.0, p=4.0):
    """kind: 'sphere' or 'rounded_cube'. Returns a dict of panel data."""
    v, f = icosphere(subdiv)
    if kind == "sphere":
        v = v * R
    elif kind == "rounded_cube":
        v = superellipsoid_map(v, a=a, p=p)
    else:
        raise ValueError(kind)

    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    centroids = (p0 + p1 + p2) / 3.0
    cross = np.cross(p1 - p0, p2 - p0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2.0 * areas[:, None])
    outward = np.sum(normals * centroids, axis=1) < 0
    normals[outward] *= -1.0
    return dict(verts=v, faces=f, tri=(p0, p1, p2),
                centroids=centroids, areas=areas, normals=normals)


# --------------------------------------------------------------------
# --------------------------------------------------------------------
_GAUSS7 = np.array([
    [0.33333333333333, 0.33333333333333, 0.33333333333333, 0.22500000000000],
    [0.79742698535309, 0.10128650732346, 0.10128650732346, 0.12593918054483],
    [0.10128650732346, 0.79742698535309, 0.10128650732346, 0.12593918054483],
    [0.10128650732346, 0.10128650732346, 0.79742698535309, 0.12593918054483],
    [0.05971587178977, 0.47014206410511, 0.47014206410511, 0.13239415278851],
    [0.47014206410511, 0.05971587178977, 0.47014206410511, 0.13239415278851],
    [0.47014206410511, 0.47014206410511, 0.05971587178977, 0.13239415278851],
])


def assemble_M(geom):
    """Double-layer matrix M_ij = (1/4pi) INT_{T_j} (r_i - r')·n_i / |r_i - r'|^3 dS'.
    The self term (i = j) vanishes for a planar panel. Source triangles are
    integrated with a 7-point quadrature rule.
    """
    p0, p1, p2 = geom["tri"]
    c = geom["centroids"]
    n = geom["normals"]
    A = geom["areas"]
    N = len(c)
    M = np.zeros((N, N))

    bl = _GAUSS7[:, :3]          # (7,3) barycentric
    w = _GAUSS7[:, 3]            # (7,)
    qp = (bl[:, 0][None, :, None] * p0[:, None, :]
          + bl[:, 1][None, :, None] * p1[:, None, :]
          + bl[:, 2][None, :, None] * p2[:, None, :])

    for i in range(N):
        d = c[i][None, None, :] - qp                 # (N,7,3)  r_i - r'
        r = np.linalg.norm(d, axis=2)                # (N,7)
        dn = np.einsum("k,jqk->jq", n[i], d)         # (r_i-r')·n_i
        with np.errstate(divide="ignore", invalid="ignore"):
            kern = dn / r**3                         # (N,7)
        integ = A * np.einsum("jq,q->j", kern, w)    # ∫ over each T_j
        integ[i] = 0.0                               # self-panel = 0
        M[i, :] = integ / (4.0 * np.pi)
    return M


def solve_sigma(geom, Ps, eps_b, eps_m):
    """Solve (I - 2 lam M) sigma = 2/(eps_b+eps_m) * Ps * n_z for sigma."""
    lam = (eps_b - eps_m) / (eps_b + eps_m)
    M = assemble_M(geom)
    N = len(geom["centroids"])
    Aop = np.eye(N) - 2.0 * lam * M
    rhs = (2.0 / (eps_b + eps_m)) * Ps * geom["normals"][:, 2]
    sigma = np.linalg.solve(Aop, rhs)
    return sigma, M


# --------------------------------------------------------------------
# --------------------------------------------------------------------
def field_at(points, geom, sigma, chunk=2000):
    """Potential and field at arbitrary exterior points (P,3), with the panel charge
    approximated as a point charge at the centroid. Large grids are processed in
    chunks to limit memory use."""
    c = geom["centroids"]
    A = geom["areas"]
    q = sigma * A
    pts = np.atleast_2d(points)
    k = 1.0 / (4.0 * np.pi * EPS0)
    P = len(pts)
    phi = np.empty(P)
    E = np.empty((P, 3))
    for s in range(0, P, chunk):
        e = min(s + chunk, P)
        d = pts[s:e, None, :] - c[None, :, :]        # (p,N,3)
        r = np.linalg.norm(d, axis=2)                # (p,N)
        r = np.where(r < 1e-12, np.inf, r)
        phi[s:e] = k * np.sum(q[None, :] / r, axis=1)
        E[s:e] = k * np.sum((q[None, :, None] * d) / r[:, :, None]**3, axis=1)
    return phi, E


# --------------------------------------------------------------------
# --------------------------------------------------------------------
def sphere_analytic(points, Ps, eps_b, eps_m, R):
    pts = np.atleast_2d(points)
    r = np.linalg.norm(pts, axis=1)
    cth = pts[:, 2] / r
    pref = Ps * R**3 / (EPS0 * (eps_b + 2 * eps_m))
    phi = pref * cth / r**2
    Er = 2 * pref * cth / r**3
    Eth = pref * np.sqrt(np.clip(1 - cth**2, 0, None)) / r**3
    Emag = pref / r**3 * np.sqrt(1 + 3 * cth**2)
    return phi, Emag, Er, Eth


if __name__ == "__main__":
    Ps, eps_b, eps_m, R = 0.26, 6.0, 1.0, 1.0
    g = build_geometry("sphere", subdiv=3, R=R)
    sig, M = solve_sigma(g, Ps, eps_b, eps_m)
    th = np.arccos(g["centroids"][:, 2] / R)
    sig_exact = 3 * Ps * np.cos(th) / (eps_b + 2 * eps_m)
    err = np.max(np.abs(sig - sig_exact)) / np.max(np.abs(sig_exact))
    print(f"panels={len(sig)}  sigma max-rel-err = {err:.3%}")
