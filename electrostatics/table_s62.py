# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Near-surface enhancement relative to the linear continuum (Table S6.2).

Saturation-model enhancement = converged sphere enhancement at d = 0
(selfcons.run) x shape factor of Eq. S6.6 (shape_selfcons.json, run_shapes.py).
Measured enhancement, Eq. S5.9 = EF_Local(s, 0) / f(eps_s) / E_continuum(s, 0),
with f = 3 eps / (2 eps + 1) and E_continuum the surface-RMS field on the probe
shell of the measured shape (surf_rms.json). Reads booth_results.npy,
ef_data.npy, surf_rms.json and shape_selfcons.json."""
import json
import os
import numpy as np

# the self-consistent sphere solver of selfcons.py, without running its scan
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "selfcons.py")).read()
exec(_src.split('D=np.load("ef_data.npy"')[0])

EF = np.load("ef_data.npy", allow_pickle=True).item()
SR = json.load(open("surf_rms.json"))
SH = json.load(open("shape_selfcons.json"))
RHO = {"MeOH": 14.89, "EtOH": 10.31, "PrOH": 8.05, "HexOH": 4.80}   # nm^-3, Table S6.1

rows = []
for s, (er, nn) in SOLV.items():
    E_sc = run(s, 0)[0]
    E_lin = solve_multilayer(np.array([RC]), np.array([6.0, er]))(RC + 0.5e-9)
    shape = SH["cube"][s]["enh"] / SH["sphere"][s]["enh"]
    sat = E_sc / E_lin * shape
    f = 3 * er / (2 * er + 1)
    meas = EF[s]["EF"][0] / f / SR[s]["0.5"]["rms"]
    rows.append((s, er, RHO[s], E_sc / E_lin, shape, sat, meas, meas / sat))

print("Table S6.2  Near-surface enhancement relative to the linear continuum")
print(f"{'solvent':>8}{'eps_r':>7}{'rho':>8}{'sphere':>9}{'shape f':>9}"
      f"{'saturation':>12}{'measured':>10}{'residual':>10}")
for s, er, rho, es, fsh, sat, meas, res in rows:
    print(f"{s:>8}{er:>7.1f}{rho:>8.2f}{es:>9.3f}{fsh:>9.3f}{sat:>12.2f}{meas:>10.2f}{res:>10.2f}")
r = np.corrcoef([x[2] for x in rows], [x[7] for x in rows])[0, 1]
print(f"\nresidual ratio vs molecular number density: r = {r:.2f}")
json.dump({x[0]: dict(sphere=x[3], shape_factor=x[4], saturation=x[5],
                      measured=x[6], residual=x[7]) for x in rows},
          open("table_s62.json", "w"), indent=1)
