# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Boundary-element solve for an arbitrary number of interfaces. The geometric matrix
is assembled once; each iteration updates only the dielectric contrasts."""
import numpy as np
from core_shell_bem import assemble_M_multi, offset_mesh, field_core_shell, _mesh_from
from scipy.linalg import lu_factor, lu_solve
EPS0=8.8541878128e-12

def build_stack(g_core, offsets):
    """Build offset meshes for each entry in offsets (geometry units)."""
    return [g_core]+[offset_mesh(g_core,t) for t in offsets]

def assemble(geoms):
    M,C,N,Ar=assemble_M_multi(geoms)
    sizes=[len(g["centroids"]) for g in geoms]
    return dict(M=M,C=C,N=N,Ar=Ar,sizes=sizes,geoms=geoms)

def solve_layers(asm, eps_regions, Ps):
    """eps_regions has len(geoms)+1 entries, from core to bulk. Returns sigma."""
    M=asm["M"]; N=asm["N"]; sizes=asm["sizes"]; n=len(N)
    lam=np.empty(n); b=np.zeros(n); i0=0
    for j,sz in enumerate(sizes):
        ein,eout=eps_regions[j],eps_regions[j+1]
        lam[i0:i0+sz]=(eout-ein)/(eout+ein)
        if j==0: b[i0:i0+sz]=(2.0/(ein+eout))*Ps*N[i0:i0+sz,2]
        i0+=sz
    A=(2.0*lam)[:,None]*M
    A[np.diag_indices(n)]+=1.0
    from scipy.linalg import solve as lasolve
    return lasolve(A,b,overwrite_a=True,check_finite=False)

def Efield(asm, sigma, pts):
    return field_core_shell(pts, asm["geoms"], sigma)
