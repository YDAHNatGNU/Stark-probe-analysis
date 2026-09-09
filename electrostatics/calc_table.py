# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Generates the field and potential tables for all media used by the interactive
calculator."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from ferroelectric_bem import build_geometry
from accurate_bem import solve_sigma_analytic, field_analytic
from core_shell_bem import offset_mesh

Ps, A_NM, P_SHAPE, SUB = 0.26, 56.7, 5.5, 3
MEDIA = {"vacuum":1.0,"air":1.00059,"HexOH":13.0,"PrOH":20.1,"EtOH":24.5,"MeOH":33.7,"water":78.4}
NREF  = {"vacuum":1.0,"air":1.0,"HexOH":1.4178,"PrOH":1.3856,"EtOH":1.3611,"MeOH":1.3288,"water":1.333}
DS = [0.5,1,2,3,5,7.5,10,15,20,30,40,50,75,100,150,200,300,500]

g = build_geometry("rounded_cube", subdiv=SUB, a=1.0, p=P_SHAPE)
out={}
for name, em in MEDIA.items():
    sig = solve_sigma_analytic(g, Ps, 6.0, em)
    rms=[]; face=[]; emin=[]; pot=[]
    for d in DS:
        gs = offset_mesh(g, d/A_NM)
        pts=gs["centroids"]; w=gs["areas"]
        E=np.linalg.norm(field_analytic(pts,g,sig),axis=1)
        rms.append(float(np.sqrt(np.sum(w*E**2)/np.sum(w))/1e8))   # MV/cm
        face.append(float(E[np.argmax(pts[:,2])]/1e8))
        emin.append(float(E.min()/1e8))
    out[name]=dict(eps=em, n=NREF[name], rms=rms, face=face, emin=emin)
    print(f"{name:8s} d=0.5: {rms[0]:8.3f}  d=100: {rms[DS.index(100)]:8.4f} MV/cm")

from scipy.integrate import quad
def phi_axis(d_nm, em, sig, g):
    """Integrate E_z along the z axis above the face centre from d to infinity -> potential (V)."""
    def Ez(t):
        z = 1.0 + (t)/A_NM
        p = np.array([[0.0,0.0,z]])
        return field_analytic(p, g, sig)[0,2]
    val,_ = quad(Ez, d_nm, 4000.0, limit=300)
    return val*1e-9   # nm -> m

print("\nSurface potential (above the face centre, axial integration)")
for name, em in MEDIA.items():
    sig = solve_sigma_analytic(g, Ps, 6.0, em)
    ps=[]
    for d in DS:
        ps.append(float(phi_axis(d, em, sig, g)))
    out[name]["phi"]=ps
    print(f"  {name:8s} phi(0.5nm) = {ps[0]:9.3f} V,  phi(100nm) = {ps[DS.index(100)]:8.4f} V")

out["_d"]=DS
json.dump(out, open("sim_table.json","w"), indent=1)
print("\nsaved sim_table.json")
