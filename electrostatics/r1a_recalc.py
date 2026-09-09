# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Recomputes the unscreened ceiling on a single consistent basis, changing one thing
at a time from the value quoted in the original manuscript."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from ferroelectric_bem import build_geometry, sphere_analytic
from accurate_bem import solve_sigma_analytic, field_analytic
from core_shell_bem import offset_mesh
EPS0,Ps,EB,P,A_NM,d0=8.8541878128e-12,0.26,6.0,5.5,56.7,0.5
out={}
out["paper"]=Ps/(3*EPS0)
f=Ps/(EPS0*(EB+2*1.0))
pt=np.array([[0,0,1+d0/A_NM]])
_,Ea,_,_=sphere_analytic(pt,Ps,EB,1.0,1.0)
out["sphere_eb6"]=f
out["sphere_eb6_exact"]=float(Ea[0])
g=build_geometry("rounded_cube",subdiv=3,a=1.0,p=P)
sig=solve_sigma_analytic(g,Ps,EB,1.0)
out["cube_face"]=float(np.linalg.norm(field_analytic(np.array([[0,0,1+d0/A_NM]]),g,sig)))
gp=offset_mesh(g,d0/A_NM)
E=np.linalg.norm(field_analytic(gp["centroids"],g,sig),axis=1)
out["cube_rms"]=float(np.sqrt(np.sum(gp["areas"]*E**2)/np.sum(gp["areas"])))
json.dump({k:v for k,v in out.items()},open("r1a_data.json","w"),indent=1)
print("="*70); print("Unscreened ceiling on a consistent basis (eps_b = 6.0, vacuum, probe at 0.5 nm)"); print("="*70)
print(f"1) Manuscript (sphere, eps_b=1)        : {out['paper']/1e8:6.2f}  [was 97.9]")
print(f"2) Sphere, eps_b=6 (formula)         : {out['sphere_eb6']/1e8:6.2f}  [was 38.6, eps_b=5.6]")
print(f"   Sphere, eps_b=6 (analytical cross-check)   : {out['sphere_eb6_exact']/1e8:6.2f}")
print(f"3) Measured shape, face centre         : {out['cube_face']/1e8:6.2f}  [was 36.2]")
print(f"4) Measured shape, surface RMS        : {out['cube_rms']/1e8:6.2f}  [was 39.1]")
print()
print("Change at each step (one variable at a time):")
print(f"  eps_b 1 -> 6   : x{out['sphere_eb6']/out['paper']:.3f}  ({100*(out['sphere_eb6']/out['paper']-1):+.1f}%)")
print(f"  sphere -> measured shape  : x{out['cube_face']/out['sphere_eb6']:.3f}  ({100*(out['cube_face']/out['sphere_eb6']-1):+.1f}%)")
print(f"  face centre -> RMS  : x{out['cube_rms']/out['cube_face']:.3f}  ({100*(out['cube_rms']/out['cube_face']-1):+.1f}%)")
