# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Decomposes the effect of the silica shell into a purely geometric and a purely
dielectric contribution by changing one variable at a time."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from ferroelectric_bem import build_geometry
from accurate_bem import solve_sigma_analytic, field_analytic
from core_shell_bem import offset_mesh
Ps,EB,A_NM,P,SUB,LINK=0.26,6.0,56.7,5.5,3,0.5
SOLV={"MeOH":33.7,"EtOH":24.5,"PrOH":20.1,"HexOH":13.0}
g0=build_geometry("rounded_cube",subdiv=SUB,a=1.0,p=P)
def rms_on(gp,fn):
    E=np.linalg.norm(fn(gp["centroids"]),axis=1)
    return float(np.sqrt(np.sum(gp["areas"]*E**2)/np.sum(gp["areas"])))
rb=json.load(open("run_b.json"))
out={}
for s,em in SOLV.items():
    sig=solve_sigma_analytic(g0,Ps,EB,em)
    gp0=offset_mesh(g0,LINK/A_NM)
    b1=rms_on(gp0,lambda p: field_analytic(p,g0,sig))/1e8
    b2=rb["bare"][s]["10.0"]/1e8
    b3=rb["shell"][s]["10.0"]/1e8
    out[s]={"ref":b1,"geometric":b2,"full":b3,
            "f_geo":b2/b1,"f_diel":b3/b2,"f_tot":b3/b1}
    print(f"{s:>6}: ref {b1:6.3f} -> geom {b2:6.3f} (x{b2/b1:.3f}) "
          f"-> total {b3:6.3f} (x{b3/b2:.3f})   total x{b3/b1:.3f}")
json.dump(out,open("r3a_data.json","w"),indent=1)
print("\nThe hard-coded values 8.38 / 6.58 / 6.29 were of unknown origin and are replaced above")
