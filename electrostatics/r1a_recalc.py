# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Unscreened field ceiling (vacuum, eps_m = 1) on a consistent basis: surface RMS of
|E| on the probe shell 0.5 nm outside the particle (Table S3.2)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from ferroelectric_bem import build_geometry, sphere_analytic
from accurate_bem import solve_sigma_analytic, field_analytic
from core_shell_bem import offset_mesh
EPS0,Ps,EB,P,A_NM,d0=8.8541878128e-12,0.26,6.0,5.5,56.7,0.5
out={}
k_shell=(A_NM/(A_NM+d0))**3        # sphere of radius a = 56.7 nm, probe shell at +0.5 nm
out["sphere_eb1_rms"]=np.sqrt(2)*Ps/(EPS0*(1.0+2))*k_shell   # Eq. S3.9 with eps_b = 1
out["sphere_eb6_rms"]=np.sqrt(2)*Ps/(EPS0*(EB+2))*k_shell    # Eq. S3.9 with eps_b = 6
g=build_geometry("rounded_cube",subdiv=3,a=1.0,p=P)
sig=solve_sigma_analytic(g,Ps,EB,1.0)
out["cube_face"]=float(np.linalg.norm(field_analytic(np.array([[0,0,1+d0/A_NM]]),g,sig)))
gp=offset_mesh(g,d0/A_NM)
E=np.linalg.norm(field_analytic(gp["centroids"],g,sig),axis=1)
out["cube_rms"]=float(np.sqrt(np.sum(gp["areas"]*E**2)/np.sum(gp["areas"])))
json.dump({k:float(v) for k,v in out.items()},open("r1a_data.json","w"),indent=1)
print("="*70); print("Table S3.2  Unscreened ceiling (vacuum), surface RMS on the probe shell [MV/cm]"); print("="*70)
print(f"  Sphere, eps_b = 1                    : {out['sphere_eb1_rms']/1e8:6.1f}")
print(f"  Sphere, eps_b = 6                    : {out['sphere_eb6_rms']/1e8:6.1f}")
print(f"  Measured shape, p = 5.5, eps_b = 6   : {out['cube_rms']/1e8:6.1f}")
print(f"  (single point, measured shape, face centre: {out['cube_face']/1e8:.1f})")
print()
print(f"  eps_b 1 -> 6            : x{out['sphere_eb6_rms']/out['sphere_eb1_rms']:.3f}")
print(f"  sphere -> measured shape: x{out['cube_rms']/out['sphere_eb6_rms']:.3f}")
print(f"  ceiling ratio (eps_b = 1 sphere / measured shape): {out['sphere_eb1_rms']/out['cube_rms']:.2f}")
