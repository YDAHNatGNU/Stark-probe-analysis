# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Parses the KPFM line profiles and extracts the step-height contrast for each
region and lift height."""
import numpy as np, openpyxl, json
LIFT=[0,5,10,15,20]
wb=openpyxl.load_workbook('../data/KPFM_raw.xlsx',data_only=True)

def col(ws,c,r0):
    v=[]
    for r in ws.iter_rows(min_row=r0,values_only=True):
        x=r[c] if c<len(r) else None
        if isinstance(x,(int,float)): v.append(float(x))
        elif v: break
    return np.array(v)

DATA={}
ws=wb["tBTO"]
DATA["tBTO"]={}
for sp,(cx,r0) in {1:(1,4),2:(11,6)}.items():
    d={}
    for i,z in enumerate(LIFT):
        x=col(ws,cx+2*i,r0); y=col(ws,cx+2*i+1,r0)
        n=min(len(x),len(y)); d[z]=(x[:n],y[:n])
    DATA["tBTO"][sp]=d
for sp,(cd,c0,r0) in {3:(21,22,4),4:(27,28,4),5:(33,34,4)}.items():
    x=col(ws,cd,r0); d={}
    for i,z in enumerate(LIFT):
        y=col(ws,c0+i,r0); n=min(len(x),len(y)); d[z]=(x[:n],y[:n])
    DATA["tBTO"][sp]=d
for name in ("cBTO","SiO2"):
    ws=wb[name]; DATA[name]={}
    for sp in range(1,6):
        cd=1+6*(sp-1); c0=cd+1
        if cd>=ws.max_column: continue
        x=col(ws,cd,5)
        if len(x)<50: continue
        d={}
        ok=True
        for i,z in enumerate(LIFT):
            y=col(ws,c0+i,5)
            if len(y)<50: ok=False; break
            n=min(len(x),len(y)); d[z]=(x[:n],y[:n])
        if ok: DATA[name][sp]=d

for k,v in DATA.items():
    print(f"{k}: spot {sorted(v)} " + " ".join(f"(s{s}:{len(v[s][0][0])}pt)" for s in sorted(v)))

def contrast(x,y):
    """Step height: upper plateau minus substrate baseline. The two levels are
    separated by intensity, which works regardless of the scan length."""
    k=9; sm=np.convolve(y,np.ones(k)/k,mode="same")
    lo=np.percentile(sm,20); hi=np.percentile(sm,98)
    amp=hi-lo
    if amp<=0: return None
    top_m = sm > lo+0.80*amp
    base_m= sm < lo+0.20*amp
    if top_m.sum()<5 or base_m.sum()<15: return None
    top=np.median(y[top_m]); base=np.median(y[base_m])
    ipk=int(np.argmax(sm[k:-k]))+k; half=(sm[ipk]+base)/2
    L=ipk
    while L>0 and sm[L]>half: L-=1
    Rr=ipk
    while Rr<len(sm)-1 and sm[Rr]>half: Rr+=1
    w=float(x[Rr]-x[L])
    return dict(contrast=float(top-base),top=float(top),base=float(base),width=w,
                xpk=float(x[ipk]),
                se=float(np.sqrt(np.var(y[top_m])/top_m.sum()+np.var(y[base_m])/base_m.sum())))

R={}
print("\n"+"="*80)
print(f"{'sample':>6} {'spot':>5} " + " ".join(f"{'z='+str(z):>11}" for z in LIFT) + f"{'width(um)':>9}")
print("="*80)
for name in ("tBTO","cBTO","SiO2"):
    R[name]={}
    for sp in sorted(DATA[name]):
        row=f"{name:>6} {sp:>5} "; rec={}
        ws=[]
        for z in LIFT:
            x,y=DATA[name][sp][z]; c=contrast(x,y)
            if c: rec[z]=c; ws.append(c["width"]); row+=f"{c['contrast']:11.1f}"
            else: row+=f"{'—':>11}"
        row+=f"{np.mean(ws) if ws else np.nan:9.2f}"
        print(row); R[name][sp]=rec
json.dump({n:{str(s):{str(z):v for z,v in r.items()} for s,r in R[n].items()} for n in R},
          open("kpfm_v1.json","w"),indent=1)
