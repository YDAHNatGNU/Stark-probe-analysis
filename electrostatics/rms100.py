# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Extends the linear-continuum calculation to distances of up to 200 nm."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from ferroelectric_bem import build_geometry
from accurate_bem import solve_sigma_analytic, field_analytic
from core_shell_bem import offset_mesh

Ps, A_NM, P_SHAPE, SUB = 0.26, 56.7, 5.5, 3
SOLV = {"MeOH":33.7,"EtOH":24.5,"PrOH":20.1,"HexOH":13.0}
D_LIST = [0.5, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 150.0, 200.0]

g = build_geometry("rounded_cube", subdiv=SUB, a=1.0, p=P_SHAPE)
out={}
for sname, em in SOLV.items():
    sig = solve_sigma_analytic(g, Ps, 6.0, em)
    rec={}
    for d in D_LIST:
        gs = offset_mesh(g, d/A_NM)
        pts=gs["centroids"]; w=gs["areas"]
        E=np.linalg.norm(field_analytic(pts,g,sig),axis=1)
        rec[d]=dict(rms=float(np.sqrt(np.sum(w*E**2)/np.sum(w))),
                    face=float(E[np.argmax(pts[:,2])]),
                    emin=float(E.min()), emax=float(E.max()))
    out[sname]=rec
json.dump(out,open("surf_rms_far.json","w"),indent=1)

print("="*84)
print("Linear-continuum surface-RMS field, extended distance range [MV/cm]")
print("="*84)
print(f"{'d(nm)':>7}"+"".join(f"{s:>11}" for s in SOLV)+f"{'rel. to d=0':>11}{'eff. exponent':>9}")
prev=None
for i,d in enumerate(D_LIST):
    row=f"{d:7.1f}"
    for s in SOLV: row+=f"{out[s][d]['rms']/1e8:11.4f}"
    frac=out['MeOH'][d]['rms']/out['MeOH'][0.5]['rms']
    row+=f"{frac*100:10.2f}%"
    if prev is not None:
        r1=(56.7+0.5+prev)*1e-9; r2=(56.7+0.5+d)*1e-9
        k=np.log(out['MeOH'][d]['rms']/out['MeOH'][prev]['rms'])/np.log(r2/r1)
        row+=f"{k:9.2f}"
    print(row); prev=d
print()
print("── Summary at d = 100 nm ──")
for s in SOLV:
    r=out[s][100.0]
    print(f"  {s:6s} RMS {r['rms']/1e8:7.4f} | face centre {r['face']/1e8:7.4f} | "
          f"range {r['emin']/1e8:.4f}–{r['emax']/1e8:.4f} MV/cm")
print(f"\n  HexOH/MeOH contrast @100nm = {out['HexOH'][100.0]['rms']/out['MeOH'][100.0]['rms']:.3f}"
      f"   (@0.5nm = {out['HexOH'][0.5]['rms']/out['MeOH'][0.5]['rms']:.3f})")
print(f"  saturation threshold E_c = 4.1 MV/cm contrast: {out['HexOH'][100.0]['rms']/1e8/4.1:.3f}x (HexOH)")
