# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Exports the extracted contrast values to spreadsheet form."""
import numpy as np, openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
import os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
exec(open(_os.path.join(_here, "kpfm_v1.py")).read().split("def contrast")[0])   # DATA, LIFT
NM=["tBTO","cBTO","SiO2"]; LBL={"SiO2":"SiO2"}

def extract(x,y,q_top=0.80,q_base=0.20):
    k=9; sm=np.convolve(y,np.ones(k)/k,mode="same")
    lo,hi=np.percentile(sm,20),np.percentile(sm,98); amp=hi-lo
    if amp<=0: return None
    tm=sm>lo+q_top*amp; bm=sm<lo+q_base*amp
    if tm.sum()<5 or bm.sum()<15: return None
    top,base=np.median(y[tm]),np.median(y[bm])
    mad=lambda v:1.4826*np.median(np.abs(v-np.median(v)))
    ipk=int(np.argmax(sm[k:-k]))+k; half=(sm[ipk]+base)/2
    L=ipk
    while L>0 and sm[L]>half: L-=1
    R=ipk
    while R<len(sm)-1 and sm[R]>half: R+=1
    return dict(peak=float(top),base=float(base),contrast=float(top-base),
                width=float(x[R]-x[L]),npk=int(tm.sum()),nbs=int(bm.sum()),
                mad_pk=float(mad(y[tm])),mad_bs=float(mad(y[bm])))

ROWS=[]
for n in NM:
    for sp in sorted(DATA[n]):
        for z in LIFT:
            r=extract(*DATA[n][sp][z])
            if r: ROWS.append((n,sp,z,r["peak"],r["base"],r["contrast"],r["width"],
                               r["npk"],r["nbs"],r["mad_pk"],r["mad_bs"]))

HDR=Font(bold=True,size=10); FILL=PatternFill("solid",fgColor="E8EEF7")
THIN=Border(bottom=Side(style="thin",color="BBBBBB"))
wb=openpyxl.Workbook(); wb.remove(wb.active)

ws=wb.create_sheet("Contrast")
ws["A1"]="Step-height contrast (mV) — peak level minus substrate level"
ws["A1"].font=Font(bold=True,size=12)
ws["A2"]="Extraction: 9-point smoothing; points above the top 20% of the intensity range are the peak, "\
         "below the bottom 20% the substrate; median of each group."
ws["A2"].font=Font(size=9,italic=True)
hd=["Sample","Spot","Width (µm)"]+[f"{z} nm" for z in LIFT]+["mean","s.d."]
for j,h in enumerate(hd):
    c=ws.cell(4,j+1,h); c.font=HDR; c.fill=FILL; c.alignment=Alignment(horizontal="center")
r=5
for n in NM:
    for sp in sorted(DATA[n]):
        v=[x for x in ROWS if x[0]==n and x[1]==sp]
        if not v: continue
        cs={x[2]:x[5] for x in v}; w=np.mean([x[6] for x in v])
        ws.cell(r,1,n); ws.cell(r,2,sp); ws.cell(r,3,round(float(w),3))
        vals=[]
        for j,z in enumerate(LIFT):
            if z in cs: ws.cell(r,4+j,round(cs[z],2)); vals.append(cs[z])
        ws.cell(r,4+len(LIFT),round(float(np.mean(vals)),2))
        ws.cell(r,5+len(LIFT),round(float(np.std(vals,ddof=1)),2))
        for j in range(len(hd)): ws.cell(r,j+1).border=THIN
        r+=1
    r+=1
for j,w in enumerate((9,6,11,10,10,10,10,10,9,8)):
    ws.column_dimensions[openpyxl.utils.get_column_letter(j+1)].width=w

ws=wb.create_sheet("z0")
ws["A1"]="Values at z = 0 (used for the sample comparison)"; ws["A1"].font=Font(bold=True,size=12)
hd=["Sample","Spot","Peak (mV)","Substrate (mV)","Contrast (mV)","Width (µm)",
    "n peak pts","n substrate pts","MAD peak","MAD substrate"]
for j,h in enumerate(hd):
    c=ws.cell(3,j+1,h); c.font=HDR; c.fill=FILL; c.alignment=Alignment(horizontal="center")
r=4
for n in NM:
    for x in [q for q in ROWS if q[0]==n and q[2]==0]:
        for j,v in enumerate([x[0],x[1],round(x[3],2),round(x[4],2),round(x[5],2),
                              round(x[6],3),x[7],x[8],round(x[9],2),round(x[10],2)]):
            ws.cell(r,j+1,v); ws.cell(r,j+1).border=THIN
        r+=1
    r+=1
r+=1
ws.cell(r,1,"Sample means at z = 0").font=Font(bold=True,size=11); r+=1
for j,h in enumerate(["Sample","n spots","Contrast mean","s.d.","s.e.m.","median","Width mean (µm)"]):
    c=ws.cell(r,j+1,h); c.font=HDR; c.fill=FILL
r+=1
for n in NM:
    c=np.array([x[5] for x in ROWS if x[0]==n and x[2]==0])
    w=np.array([x[6] for x in ROWS if x[0]==n and x[2]==0])
    for j,v in enumerate([n,len(c),round(float(c.mean()),2),round(float(c.std(ddof=1)),2),
                          round(float(c.std(ddof=1)/np.sqrt(len(c))),2),
                          round(float(np.median(c)),2),round(float(w.mean()),3)]):
        ws.cell(r,j+1,v)
    r+=1
for j,w in enumerate((9,6,12,14,13,11,11,15,11,14)):
    ws.column_dimensions[openpyxl.utils.get_column_letter(j+1)].width=w

ws=wb.create_sheet("All values")
hd=["Sample","Spot","Lift (nm)","Peak (mV)","Substrate (mV)","Contrast (mV)","Width (µm)",
    "n peak","n substrate","MAD peak","MAD substrate"]
for j,h in enumerate(hd):
    c=ws.cell(1,j+1,h); c.font=HDR; c.fill=FILL
for i,x in enumerate(ROWS):
    for j,v in enumerate(x):
        ws.cell(2+i,j+1, round(v,3) if isinstance(v,float) else v)
for j,w in enumerate((9,6,10,11,14,13,11,9,12,11,14)):
    ws.column_dimensions[openpyxl.utils.get_column_letter(j+1)].width=w

wb.save("KPFM_extracted_values.xlsx")
print(f"saved — {len(ROWS)} rows")
for n in NM:
    c=[x[5] for x in ROWS if x[0]==n and x[2]==0]
    print(f"  {n}: contrast at z=0 " + ", ".join(f"{v:.1f}" for v in c))
