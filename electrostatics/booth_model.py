# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Field-dependent permittivity (dielectric saturation) model.

1) The correlated dipole moment of each alcohol is obtained by inverting the
   Kirkwood-Frohlich relation from tabulated permittivity, refractive index and
   number density, so no literature value of the Kirkwood factor is assumed.
2) Booth-type Langevin saturation.
3) Self-consistent solution at each distance.
4) Half-decay distances are extracted and compared with the measurement.

Note: n, rho and mu are standard tabulated values at room temperature; their
sources are stated in the paper."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from scipy.optimize import brentq
from scipy.interpolate import interp1d

EPS0 = 8.8541878128e-12
KB = 1.380649e-23
T = 298.15
DEBYE = 3.33564e-30
NA = 6.02214076e23

SOLV = {
    #        eps_bulk,  n,      rho g/cm3, M g/mol, mu_gas (D)
    "MeOH":  (33.7, 1.3288, 0.792, 32.04, 1.70),
    "EtOH":  (24.5, 1.3611, 0.789, 46.07, 1.69),
    "PrOH":  (20.1, 1.3856, 0.803, 60.10, 1.68),
    "HexOH": (13.0, 1.4178, 0.814, 102.17, 1.65),
}
# Measured half-decay distances, Eq. S5.8: log-linear interpolation between the
# two separations that bracket E(0)/2, applied to the measured EF_Local(d)
# (ef_data.npy). Uncertainty: 20,000 Monte Carlo draws of EF within its s.d.
_EF = np.load("ef_data.npy", allow_pickle=True).item()
_DMEAS = np.array([0.0, 5.0, 10.0, 15.0, 20.0])


def half_decay(E, d=_DMEAS):
    y = E / E[0]
    i = np.where(y < 0.5)[0]
    if len(i) == 0:
        return np.nan
    i = i[0]
    if E[i] > 0 and E[i-1] > 0:
        return d[i-1] + (d[i]-d[i-1])*np.log(2*E[i-1]/E[0])/np.log(E[i-1]/E[i])
    return np.interp(0.5, [y[i], y[i-1]], [d[i], d[i-1]])


MEAS_HALF, MEAS_HALF_SD = {}, {}
_rng = np.random.default_rng(0)
for _s in ("MeOH", "EtOH", "PrOH", "HexOH"):
    _E, _sE = _EF[_s]["EF"], np.nan_to_num(_EF[_s]["sEF"])
    MEAS_HALF[_s] = float(half_decay(_E))
    _mc = np.array([half_decay(_E + _rng.normal(0, 1, len(_E))*_sE) for _ in range(20000)])
    MEAS_HALF_SD[_s] = float(np.nanstd(_mc))

sr = json.load(open("surf_rms.json"))
_d = [0.5, 5.0, 10.0, 15.0, 20.0]
_G = np.mean([[sr[s][str(x)]["rms"] for x in _d] for s in sr], axis=0)
_G = _G/_G[0]
Gf = interp1d(np.concatenate([[0.0], _d]), np.concatenate([[1.0], _G]),
              kind="cubic", fill_value="extrapolate")
C_RMS, EB_EFF = 2.1336e10, 4.456      # E_rms = C/(eb_eff + eps_m)


def number_density(rho, M):
    """Molecular number density (m^-3)."""
    return rho*1e3/(M*1e-3)*NA/1.0 / 1e3 * 1e3 / 1.0 * 1.0 if False else \
        (rho*1e3)/(M*1e-3)*NA


def kirkwood_gmu2(eps, n, rho, M):
    """Invert the Kirkwood-Frohlich relation for g_k * mu^2 (SI units).
       (eps-n^2)(2eps+n^2)/(eps(n^2+2)^2) = N g mu^2/(9 eps0 kB T)"""
    N = number_density(rho, M)
    lhs = (eps-n**2)*(2*eps+n**2)/(eps*(n**2+2)**2)
    return lhs*9*EPS0*KB*T/N, N


def langevin(x):
    x = np.asarray(x, float)
    out = np.where(np.abs(x) < 1e-6, x/3.0,
                   1.0/np.tanh(np.where(np.abs(x) < 1e-6, 1.0, x)) - 1.0/np.where(np.abs(x) < 1e-6, 1.0, x))
    return out


def eps_of_E(E, eps_bulk, n, mu_eff):
    """Booth-type saturation with x = mu_eff * E / (kB T)."""
    x = mu_eff*np.abs(E)/(KB*T)
    x = np.maximum(x, 1e-9)
    return n**2 + (eps_bulk - n**2)*3.0*langevin(x)/x


print("="*78)
print("1) Correlated dipole from the Kirkwood-Frohlich inversion")
print("="*78)
print(f"{'solvent':>7}{'eps':>7}{'n':>8}{'N (1/nm3)':>12}{'sqrt(g)*mu (D)':>16}"
      f"{'g_k':>8}{'E_c (MV/cm)':>13}")
PAR = {}
for s, (eps, n, rho, M, mu) in SOLV.items():
    gmu2, N = kirkwood_gmu2(eps, n, rho, M)
    mu_eff = np.sqrt(gmu2)                      # sqrt(g_k)*mu
    g_k = gmu2/(mu*DEBYE)**2
    E_c = KB*T/mu_eff
    PAR[s] = dict(eps=eps, n=n, mu_eff=mu_eff, E_c=E_c, g_k=g_k, N=N)
    print(f"{s:>7}{eps:>7.1f}{n:>8.4f}{N/1e27:>12.2f}{mu_eff/DEBYE:>16.2f}"
          f"{g_k:>8.2f}{E_c/1e8:>13.2f}")

print("\n" + "="*78)
print("2) Self-consistent solution:  E(d) = C*G(d)/(eb_eff + eps(E))")
print("="*78)
dd = np.linspace(0.0, 30.0, 121)
res = {}
for s in SOLV:
    p = PAR[s]
    prof = []
    for d in dd:
        num = C_RMS*float(Gf(min(d, 20.0)))
        f = lambda E: E*(EB_EFF + eps_of_E(E, p["eps"], p["n"], p["mu_eff"])) - num
        prof.append(brentq(f, 1e5, 1e11))
    res[s] = np.array(prof)

print(f"{'d(nm)':>6}" + "".join(f"{s:>11}" for s in SOLV)
      + "   |" + "".join(f"{s+' eps':>11}" for s in SOLV))
for i in range(0, 81, 10):
    d = dd[i]
    row = "".join(f"{res[s][i]/1e8:>11.2f}" for s in SOLV)
    epsr = "".join(f"{eps_of_E(res[s][i], PAR[s]['eps'], PAR[s]['n'], PAR[s]['mu_eff']):>11.1f}"
                   for s in SOLV)
    print(f"{d:>6.1f}{row}   |{epsr}")

print("\n" + "="*78)
print("3) Half-decay distance comparison")
print("="*78)
# Values listed in Table S5.1 (Monte Carlo means, deposited as halfdecay_mc.json;
# the script that generated them was not preserved).
try:
    REPORTED = json.load(open("halfdecay_mc.json"))
except OSError:
    REPORTED = {}
print(f"{'solvent':>7}{'eps_r':>8}{'saturation':>11}{'linear':>12}{'measured (Eq. S5.8)':>22}"
      + (f"{'Table S5.1':>20}" if REPORTED else ""))
for s in SOLV:
    y = res[s]/res[s][0]
    i = np.where(y < 0.5)[0]
    hs = np.interp(0.5, [y[i[0]], y[i[0]-1]], [dd[i[0]], dd[i[0]-1]]) if len(i) else np.nan
    # linear continuum on the measured shape (surf_rms.json), same Eq. S5.8
    Ec = np.array([sr[s][str(x)]["rms"] for x in _d])
    hc = half_decay(Ec, np.array([0.0, 5.0, 10.0, 15.0, 20.0]))
    rep = (f"{REPORTED[s][0]:>8.2f} +- {REPORTED[s][1]:.2f}" if REPORTED else "")
    print(f"{s:>7}{SOLV[s][0]:>8.1f}{hs:>11.1f}{hc:>12.1f}{MEAS_HALF[s]:>12.1f} +- {MEAS_HALF_SD[s]:.1f}{rep}")
np.save("booth_results.npy", {"dd": dd, "res": res, "PAR": PAR}, allow_pickle=True)
