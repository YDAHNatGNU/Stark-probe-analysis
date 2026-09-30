# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Rebuilds the Stark fields EF_Local(d) from the linewidth table of Source Data
(Supplementary Note 2, Eq. 1) and checks that the result is identical to the
deposited data/ef_data.npy.

Each row is (shell thickness d in nm, number of spectra n, mean FWHM in eV,
FWHM standard deviation in eV); the last entry of each solvent is the
solvent-specific Stark-free baseline FWHM_0 (SiO2 control)."""
import numpy as np

E_CH, DMU_C = 1.602176634e-19, 2.267568072e-30   # (2 sqrt(2 ln2) / sqrt(3)) |dmu|, |dmu| = 0.5 D (Eq. 1)
RAW = {
 "MeOH":(33.7,[(0,21,0.0869161904761905,0.00114245120766918),
               (5,16,0.082785,0.00160124534867917),
               (10,16,0.0812325,0.00157669908352863),
               (15,16,0.079783125,0.00163511760535239),
               (20,16,0.0782525,0.00107785280380332)],0.078100625),
 "EtOH":(24.5,[(0,22,0.0894395454545455,0.00183882850199258),
               (5,19,0.0816473684210526,0.00303230356532467),
               (10,22,0.0794172727272727,0.0027714757660997),
               (15,22,0.0772704545454546,0.00244545151595429),
               (20,22,0.0753490909090909,0.00145847575062169)],0.07542),
 "PrOH":(20.1,[(0,21,0.0924095238095238,0.00212672630159707),
               (5,17,0.0799758823529412,0.00366648001975929),
               (10,19,0.0777873684210526,0.00192135323291009),
               (15,21,0.0738061904761905,0.00127550176867959),
               (20,20,0.073358,0.000893706651506727)],0.0729561904761905),
 "HexOH":(13.0,[(0,20,0.094037,0.00248543144618479),
                (5,20,0.0780575,0.00188840750229839),
                (10,21,0.0756590476190476,0.00200642195153937),
                (15,21,0.0732180952380953,0.00247471941651497),
                (20,21,0.0709104761904762,0.000711600141866739)],0.070922380952381),
}


def rebuild():
    D = {}
    for s, (eps, rows, F0) in RAW.items():
        EF, sEF, Y, sY = [], [], [], []
        for d, n, F, sd in rows:
            sem = sd / np.sqrt(n)
            dF2 = F**2 - F0**2
            Y.append(dF2); sY.append(2 * F * sem)
            if dF2 <= 0:
                EF.append(0.0); sEF.append(np.nan); continue
            e = np.sqrt(dF2) * E_CH / DMU_C
            EF.append(e); sEF.append(e * (F * sem) / dF2)
        D[s] = dict(Y=np.array(Y), sY=np.array(sY), EF=np.array(EF),
                    sEF=np.array(sEF), F0=F0)
    return D


if __name__ == "__main__":
    D = rebuild()
    np.save("ef_data_rebuilt.npy", D, allow_pickle=True)
    dep = np.load("ef_data.npy", allow_pickle=True).item()
    same = all(np.array_equal(np.nan_to_num(np.asarray(D[s][k], float)),
                              np.nan_to_num(np.asarray(dep[s][k], float)))
               for s in D for k in D[s])
    print("EF_Local(d) rebuilt from the Source Data linewidth table [MV/cm]")
    for s in D:
        print(f"  {s:6s} " + "  ".join(f"{e/1e8:6.2f}" for e in D[s]["EF"]))
    print("identical to deposited ef_data.npy:", same)
