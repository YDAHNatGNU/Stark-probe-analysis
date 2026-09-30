# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Regenerates all validation and reference data and writes them to a single JSON
file (validation_data.json)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json, gc, time
from ferroelectric_bem import (build_geometry, sphere_analytic, EPS0,
                               field_at)
from accurate_bem import (solve_sigma_analytic, solve_sigma_analytic_inplace,
                          field_analytic)
from core_shell_bem import (offset_mesh, solve_core_shell, field_core_shell,
                            analytic_E)

Ps = 0.26
OUT = {}

# ------------------------------------------------------------------
# ------------------------------------------------------------------
print("V1: sphere validation (Table S3.1) ...")
eb, em, R = 6.0, 24.5, 1.0
th = np.linspace(0.05, np.pi-0.05, 25)
rfacs = [1.05, 1.25, 1.5, 2.0, 3.0]
v1 = {"rfac": rfacs, "levels": {}}
for sub in (2, 3, 4, 5):
    g = build_geometry("sphere", subdiv=sub, R=R)
    # subdiv 5 (20,480 panels) uses the in-place solver to keep memory ~3.5 GB
    sig = (solve_sigma_analytic(g, Ps, eb, em) if sub < 5
           else solve_sigma_analytic_inplace(g, Ps, eb, em))
    errs, bem, ana = [], [], []
    for rf in rfacs:
        pts = np.column_stack([rf*np.sin(th), np.zeros_like(th), rf*np.cos(th)])
        Eb = np.linalg.norm(field_analytic(pts, g, sig), axis=1)
        _, Ea, _, _ = sphere_analytic(pts, Ps, eb, em, R)
        errs.append(float(np.max(np.abs(Eb-Ea)/Ea)))
        bem.append(float(Eb[0])); ana.append(float(Ea[0]))
    v1["levels"][sub] = {"n_panels": int(len(sig)), "max_rel_err": errs,
                         "bem_theta0": bem, "analytic_theta0": ana}
    print(f"   subdiv={sub} N={len(sig)} err={['%.4f'%e for e in errs]}")
    del g, sig; gc.collect()

g4 = build_geometry("sphere", subdiv=4, R=R)
sig4 = solve_sigma_analytic(g4, Ps, eb, em)
zl = np.linspace(1.05, 4.0, 30)
pax = np.column_stack([np.zeros_like(zl), np.zeros_like(zl), zl])
Eb = np.linalg.norm(field_analytic(pax, g4, sig4), axis=1)
_, Ea, _, _ = sphere_analytic(pax, Ps, eb, em, R)
v1["axis_profile"] = {"r_over_R": list(zl), "BEM": list(map(float, Eb)),
                      "analytic": list(map(float, Ea))}
thq = np.arccos(np.clip(g4["centroids"][:, 2]/R, -1, 1))
o = np.argsort(thq)
v1["sigma"] = {"theta_deg": list(np.degrees(thq[o])[::8]),
               "BEM": list(map(float, sig4[o][::8])),
               "analytic": list(map(float, (3*Ps*np.cos(thq[o])/(eb+2*em))[::8]))}
OUT["V1_sphere_validation"] = v1
del g4, sig4; gc.collect()

# ------------------------------------------------------------------
# ------------------------------------------------------------------
print("V2: core-shell validation (Table S4.2; fields in V/m) ...")
es, R1, t = 3.9, 1.0, 0.20
R2 = R1+t
v2 = {"levels": {}}
vals = {}
for sub in (2, 3, 4):
    gc_ = build_geometry("sphere", subdiv=sub, R=R1)
    gs_ = build_geometry("sphere", subdiv=sub, R=R2)
    sig, n1, C, N, Ar, gg = solve_core_shell(gc_, gs_, Ps, eb, es, em)
    pts = np.column_stack([2*R2*np.sin(th), np.zeros_like(th), 2*R2*np.cos(th)])
    Eb = np.linalg.norm(field_core_shell(pts, gg, sig), axis=1)
    Ea = analytic_E(pts, Ps, eb, es, em, R1, R2)
    err = float(np.max(np.abs(Eb-Ea)/Ea))
    vals[sub] = float(Eb.mean())
    v2["levels"][sub] = {"n_panels": int(len(sig)), "max_rel_err": err,
                         "mean_BEM": float(Eb.mean()),
                         "mean_analytic": float(Ea.mean())}
    print(f"   subdiv={sub} N={len(sig)} mean|E| BEM={Eb.mean()/1e6:.2f} MV/m "
          f"analytic={Ea.mean()/1e6:.2f} MV/m  max rel err={err:.4f}")
    del gc_, gs_, sig, gg; gc.collect()
k = np.log2(abs((vals[2]-vals[3])/(vals[3]-vals[4])))
Einf = vals[4] + (vals[4]-vals[3])/(2**k-1)
v2["richardson"] = {"order_k": float(k), "extrapolated": float(Einf),
                    "analytic": float(Ea.mean()),
                    "rel_err": float(abs(Einf-Ea.mean())/Ea.mean())}
print(f"   Richardson k={k:.2f} E_inf={Einf/1e6:.2f} MV/m err={v2['richardson']['rel_err']:.4f}")
OUT["V2_coreshell_validation"] = v2

# ------------------------------------------------------------------
# ------------------------------------------------------------------
print("V3: shape sensitivity and ceiling ...")
d0 = 0.5/56.7
v3 = {"p_scan": [], "ceiling": []}
for p in (2.0, 3.0, 4.0, 5.0, 5.5, 6.0, 7.0, 10.0):
    g = build_geometry("rounded_cube", subdiv=3, a=1.0, p=p)
    sig = solve_sigma_analytic(g, Ps, 6.0, 1.0)
    E = float(np.linalg.norm(field_analytic(np.array([[0, 0, 1+d0]]), g, sig)))
    v3["p_scan"].append({"p": p, "E_vacuum_face": E})
    print(f"   p={p}: {E/1e8:.2f} MV/cm")
    del g, sig; gc.collect()

v3["ceiling"] = [
    {"step": "Paper: sphere, eps_b=1", "value": float(Ps/(3*EPS0))},
    {"step": "physical eps_b=5.6, sphere equator", "value": float(Ps/(EPS0*(5.6+2)))},
    {"step": "measured shape p=5.5, face centre",
     "value": float([x["E_vacuum_face"] for x in v3["p_scan"] if x["p"] == 5.5][0])},
]
OUT["V3_shape_ceiling"] = v3

json.dump(OUT, open("validation_data.json", "w"), indent=1)
print("\nsaved validation_data.json")
