# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Reproduces the KPFM values reported in the paper from the regions analysed
there: the step-height contrast at z = 0 (mean +- s.e.m.), unpaired two-sided
Student t-tests between samples, and the relative decay rate of the contrast over lift heights of
0-20 nm (per-region linear fit, -slope/intercept x 100). Reads kpfm_v1.json,
written by kpfm_v1.py. kpfm_z0.py gives the corresponding summary over all
regions in KPFM_raw.xlsx."""
import json
import numpy as np
from scipy import stats

LIFT = [0, 5, 10, 15, 20]
# Regions (numbered as in KPFM_raw.xlsx) analysed in the paper.
PAPER_REGIONS = {"tBTO": [3, 4, 5], "cBTO": [1, 3, 4], "SiO2": [2, 4, 5]}
PAIRS = [("tBTO", "cBTO"), ("tBTO", "SiO2"), ("cBTO", "SiO2")]


def sem(v):
    return v.std(ddof=1) / np.sqrt(len(v))


R = json.load(open("kpfm_v1.json"))
z = np.array(LIFT, float)
C, REL = {}, {}
for name, regions in PAPER_REGIONS.items():
    C[name] = np.array([[R[name][str(r)][str(l)]["contrast"] for l in LIFT]
                        for r in regions])
    fits = [stats.linregress(z, c) for c in C[name]]
    REL[name] = np.array([-f.slope / f.intercept * 100 for f in fits])

print("Contrast at z = 0, regions analysed in the paper (mean +- s.e.m.)")
for name, regions in PAPER_REGIONS.items():
    c0 = C[name][:, 0]
    print(f"  {name:5} {c0.mean():5.1f} +- {sem(c0):.1f} mV   regions {regions}")
print("Unpaired two-sided Student t-tests, contrast at z = 0")
for a, b in PAIRS:
    p = stats.ttest_ind(C[a][:, 0], C[b][:, 0], equal_var=True).pvalue
    print(f"  {a} vs {b}: p = {p:.3f}")
print("Relative decay rate over 0-20 nm (mean +- s.e.m.)")
for name in PAPER_REGIONS:
    print(f"  {name:5} {REL[name].mean():5.2f} +- {sem(REL[name]):.2f} %/nm")
print("Unpaired two-sided Student t-tests, relative decay rate")
for a, b in PAIRS:
    p = stats.ttest_ind(REL[a], REL[b], equal_var=True).pvalue
    print(f"  {a} vs {b}: p = {p:.3f}")
