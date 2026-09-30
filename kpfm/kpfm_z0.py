# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Analysis at zero nominal lift: robust estimation, hierarchical bootstrap over
regions, and sensitivity to the definition of the peak and substrate levels."""
import numpy as np, json
from scipy import stats
import os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
exec(open(_os.path.join(_here, "kpfm_v1.py")).read().split("def contrast")[0])   # DATA, LIFT
NM=["tBTO","cBTO","SiO2"]
rng=np.random.default_rng(0)

def extract(x,y,q_top=0.80,q_base=0.20,plo=20,phi=98):
    """Separate the peak and substrate levels by intensity; return medians and MADs."""
    k=9; sm=np.convolve(y,np.ones(k)/k,mode="same")
    lo,hi=np.percentile(sm,plo),np.percentile(sm,phi); amp=hi-lo
    if amp<=0: return None
    tm=sm>lo+q_top*amp; bm=sm<lo+q_base*amp
    if tm.sum()<5 or bm.sum()<15: return None
    top,base=np.median(y[tm]),np.median(y[bm])
    mad=lambda v: 1.4826*np.median(np.abs(v-np.median(v)))
    ipk=int(np.argmax(sm[k:-k]))+k; half=(sm[ipk]+base)/2
    L=ipk
    while L>0 and sm[L]>half: L-=1
    R=ipk
    while R<len(sm)-1 and sm[R]>half: R+=1
    return dict(c=float(top-base),top=float(top),base=float(base),
                mad_top=float(mad(y[tm])),mad_base=float(mad(y[bm])),
                n_top=int(tm.sum()),n_base=int(bm.sum()),width=float(x[R]-x[L]))

print("="*82); print("Extraction from the z = 0 profiles (median +- MAD)"); print("="*82)
Z0={}
for n in NM:
    Z0[n]=[]
    for sp in sorted(DATA[n]):
        r=extract(*DATA[n][sp][0])
        if r is None: continue
        Z0[n].append(r)
        print(f"  {n:>6} spot{sp}: contrast {r['c']:5.1f} mV | peak {r['top']:7.1f}±{r['mad_top']:4.1f} "
              f"({r['n_top']:3d}pt) | substrate {r['base']:7.1f}±{r['mad_base']:4.1f} ({r['n_base']:3d}pt) "
              f"| width {r['width']:.2f} µm")

print("\n"+"="*82); print("Error sources: within-profile noise versus region-to-region variation"); print("="*82)
for n in NM:
    c=np.array([r["c"] for r in Z0[n]])
    inn=np.mean([np.sqrt(r["mad_top"]**2/r["n_top"]+r["mad_base"]**2/r["n_base"]) for r in Z0[n]])
    btw=c.std(ddof=1)
    print(f"  {n:>6}: within-profile s.e. {inn:5.2f} mV | between-region s.d. {btw:5.2f} mV | ratio {btw/inn:5.1f}x")
print("  -> region-to-region variation dominates; the bootstrap must resample regions")

print("\n"+"="*82); print("Hierarchical bootstrap over regions (20000 resamples)"); print("="*82)
BOOT={}
for n in NM:
    c=np.array([r["c"] for r in Z0[n]])
    mads=np.array([np.sqrt(r["mad_top"]**2/r["n_top"]+r["mad_base"]**2/r["n_base"]) for r in Z0[n]])
    out=np.empty(20000)
    for i in range(20000):
        idx=rng.integers(0,len(c),len(c))
        out[i]=(c[idx]+rng.normal(0,mads[idx])).mean()
    BOOT[n]=out
    lo,hi=np.percentile(out,[2.5,97.5])
    print(f"  {n:>6}: {c.mean():5.1f} mV  95% CI [{lo:5.1f}, {hi:5.1f}]  (n={len(c)} spots)")
print()
for a,b in (("tBTO","cBTO"),("tBTO","SiO2"),("cBTO","SiO2")):
    d=BOOT[a]-BOOT[b]; lo,hi=np.percentile(d,[2.5,97.5])
    p=2*min((d>0).mean(),(d<0).mean())
    print(f"  {a} − {b} = {d.mean():+5.1f} mV  95% CI [{lo:+5.1f}, {hi:+5.1f}]  p = {p:.3f}")

print("\n"+"="*82); print("Definition sensitivity: varying the level thresholds"); print("="*82)
print(f"{'q_top / q_base':>16} " + "".join(f"{n:>10}" for n in NM) + f"{'t−c':>9}{'t−S':>9}")
for qt,qb in ((0.80,0.20),(0.90,0.10),(0.70,0.30),(0.85,0.15),(0.75,0.25)):
    row=f"{qt:.2f} / {qb:.2f}      "
    mv={}
    for n in NM:
        v=[extract(*DATA[n][sp][0],q_top=qt,q_base=qb) for sp in sorted(DATA[n])]
        v=[r["c"] for r in v if r]
        mv[n]=np.mean(v); row+=f"{np.mean(v):10.1f}"
    row+=f"{mv['tBTO']-mv['cBTO']:9.1f}{mv['tBTO']-mv['SiO2']:9.1f}"
    print(row)

json.dump({n:[{k:float(v) for k,v in r.items()} for r in Z0[n]] for n in NM},
          open("kpfm_z0.json","w"),indent=1)
