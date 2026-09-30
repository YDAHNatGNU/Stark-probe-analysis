# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Step-height contrast of one tBTO and one cBTO region across lift heights.
Descriptive only: with a single region per sample, the lift heights are not
independent replicates, so no significance test is made here (see
kpfm_paper_values.py and kpfm_z0.py for between-region statistics)."""
import os as _os
# Input data are read from the data folder beside the analysis folders;
# the location can be overridden with the DATA_DIR environment variable.
_DATA = _os.environ.get("DATA_DIR") or _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "data")

import numpy as np, openpyxl, json
LIFT=[0,5,10,15,20]
def load(path,sheet=None):
    wb=openpyxl.load_workbook(path,data_only=True)
    ws=wb[sheet] if sheet else wb[wb.sheetnames[0]]
    rows=[r for r in ws.iter_rows(min_row=6,values_only=True)]
    P={}
    for i,z in enumerate(LIFT):
        x=[];y=[]
        for r in rows:
            a,b=r[2*i],r[2*i+1]
            if a is None or b is None: continue
            x.append(float(a)); y.append(float(b))
        P[z]=(np.array(x),np.array(y))
    return P

def contrast(P):
    out={}
    for z in LIFT:
        x,y=P[z]; k=9
        sm=np.convolve(y,np.ones(k)/k,mode="same")
        ipk=int(np.argmax(sm[k:-k]))+k; xpk=x[ipk]
        base0=np.median(np.r_[sm[:80],sm[-80:]])
        half=(sm[ipk]+base0)/2
        L=ipk
        while L>0 and sm[L]>half: L-=1
        Rr=ipk
        while Rr<len(sm)-1 and sm[Rr]>half: Rr+=1
        w=x[Rr]-x[L]
        m=(x>xpk-0.25*w)&(x<xpk+0.25*w); mb=(np.abs(x-xpk)>1.5*w)
        top=np.median(y[m]); base=np.median(y[mb])
        out[z]=dict(top=float(top),base=float(base),contrast=float(top-base),
            se=float(np.sqrt(np.var(y[m])/m.sum()+np.var(y[mb])/mb.sum())),
            width=float(w),xpk=float(xpk))
    return out

T=load(_os.path.join(_DATA, 'tBTO_KPFM_Raw.xlsx'))
C=load(_os.path.join(_DATA, 'cBTO_KPFM_Raw.xlsx'))
RT=contrast(T); RC=contrast(C)
json.dump({"tBTO":RT,"cBTO":RC},open("kpfm_both.json","w"),indent=1)
json.dump({"tBTO":{str(z):[T[z][0].tolist(),T[z][1].tolist()] for z in LIFT},
           "cBTO":{str(z):[C[z][0].tolist(),C[z][1].tolist()] for z in LIFT}},
          open("kpfm_prof2.json","w"))

print("="*78)
print("Step-height contrast comparison [mV]")
print("="*78)
print(f"{'z(nm)':>6} | {'tBTO peak':>11} {'substrate':>8} {'contrast':>14} | "
      f"{'cBTO peak':>11} {'substrate':>8} {'contrast':>14}")
for z in LIFT:
    t,c=RT[z],RC[z]
    print(f"{z:6d} | {t['top']:11.1f} {t['base']:8.1f} {t['contrast']:8.1f}±{t['se']:4.1f} | "
          f"{c['top']:11.1f} {c['base']:8.1f} {c['contrast']:8.1f}±{c['se']:4.1f}")
print()
print(f"{'z(nm)':>6} {'tBTO width(um)':>12} {'cBTO width(um)':>12} {'diff Δ=t−c (mV)':>18}")
for z in LIFT:
    d=RT[z]['contrast']-RC[z]['contrast']
    sd=np.sqrt(RT[z]['se']**2+RC[z]['se']**2)
    print(f"{z:6d} {RT[z]['width']:12.2f} {RC[z]['width']:12.2f} {d:10.1f}±{sd:.1f}")
print()
ct=np.array([RT[z]['contrast'] for z in LIFT]); st=np.array([RT[z]['se'] for z in LIFT])
cc=np.array([RC[z]['contrast'] for z in LIFT]); sc=np.array([RC[z]['se'] for z in LIFT])
print("── mean contrast over lift heights (single region each; within-profile s.e.) ──")
wt=1/st**2; wc=1/sc**2
mt=(wt*ct).sum()/wt.sum(); mc=(wc*cc).sum()/wc.sum()
et=1/np.sqrt(wt.sum()); ec=1/np.sqrt(wc.sum())
print(f"  tBTO {mt:6.2f} ± {et:.2f} mV")
print(f"  cBTO {mc:6.2f} ± {ec:.2f} mV")
dd=mt-mc; sdd=np.sqrt(et**2+ec**2)
print(f"  diff {dd:6.2f} ± {sdd:.2f} mV")
print(f"  ratio   {mt/mc:.2f}x")
