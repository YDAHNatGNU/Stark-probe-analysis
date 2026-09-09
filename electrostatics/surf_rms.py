# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Distribution and area-weighted RMS of |E| over the probe shell. The RMS is the
quantity that corresponds to inhomogeneous Stark broadening."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from ferroelectric_bem import build_geometry
from accurate_bem import solve_sigma_analytic, field_analytic
from core_shell_bem import offset_mesh

Ps, A_NM, P_SHAPE, SUB = 0.26, 56.7, 5.5, 3
SOLV = {"MeOH":33.7,"EtOH":24.5,"PrOH":20.1,"HexOH":13.0}
D_LIST = [0.5, 5.0, 10.0, 15.0, 20.0]   # nm

g = build_geometry("rounded_cube", subdiv=SUB, a=1.0, p=P_SHAPE)
out = {}
for sname, em in SOLV.items():
    sig = solve_sigma_analytic(g, Ps, 6.0, em)
    rec = {}
    for d in D_LIST:
        gs = offset_mesh(g, d/A_NM)
        pts = gs["centroids"]; w = gs["areas"]
        E = np.linalg.norm(field_analytic(pts, g, sig), axis=1)
        rms = np.sqrt(np.sum(w*E**2)/np.sum(w))
        mean = np.sum(w*E)/np.sum(w)
        iz = np.argmax(gs["centroids"][:,2])
        face = E[iz]
        rec[d] = dict(rms=float(rms), mean=float(mean), face=float(face),
                      emin=float(E.min()), emax=float(E.max()))
    out[sname] = rec
    print(sname, "done")
json.dump(out, open("surf_rms.json","w"), indent=1)

print("\n"+"="*78)
print("Distribution of |E| over the probe shell [MV/cm]")
print("="*78)
for s in SOLV:
    print(f"\n[{s}]  eps_r={SOLV[s]}")
    print(f"{'d(nm)':>6}{'face centre':>10}{'surface RMS':>10}{'surface mean':>10}"
          f"{'min':>9}{'max':>9}{'RMS/face centre':>12}")
    for d in D_LIST:
        r = out[s][d]
        print(f"{d:>6.1f}{r['face']/1e8:>10.2f}{r['rms']/1e8:>10.2f}"
              f"{r['mean']/1e8:>10.2f}{r['emin']/1e8:>9.2f}{r['emax']/1e8:>9.2f}"
              f"{r['rms']/r['face']:>12.3f}")
