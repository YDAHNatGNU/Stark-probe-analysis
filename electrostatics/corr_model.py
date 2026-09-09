# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Combined saturation and correlation-limited screening model. The correlation length
ratios across solvents are fixed externally, leaving a single global scale as the
only free parameter, fitted to the enhancement at zero shell thickness; the
contrast versus distance is then a free prediction."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
EPS0,Ps=8.8541878128e-12,0.26; RC=56.7e-9
SOLV={"MeOH":(33.7,1.3288),"EtOH":(24.5,1.3611),"PrOH":(20.1,1.3856),"HexOH":(13.0,1.4178)}
XIREL={"MeOH":1.0,"EtOH":0.828,"PrOH":0.732,"HexOH":0.564}
SF={"MeOH":0.845,"EtOH":0.726,"PrOH":0.681,"HexOH":0.709}
B=np.load("booth_results.npy",allow_pickle=True).item()
EC={s:B["PAR"][s]["E_c"] for s in SOLV}
bud=json.load(open("budget4.json"))
MEAS_ENH={s:bud[s]["meas"] for s in SOLV}
def Lfac(x): x=np.maximum(x,1e-6); return (3/x)*(1/np.tanh(x)-1/x)
def solve_ml(radii,eps):
    K=len(radii); n=2*K; M=np.zeros((n,n)); b=np.zeros(n)
    def T(i):
        if i==0: return [(0,lambda r:r)],[(0,lambda r:1.0)]
        if i==K: return [(n-1,lambda r:r**-2)],[(n-1,lambda r:-2*r**-3)]
        a=1+2*(i-1)
        return ([(a,lambda r:r),(a+1,lambda r:r**-2)],
                [(a,lambda r:1.0),(a+1,lambda r:-2*r**-3)])
    row=0
    for j,Rj in enumerate(radii):
        fL,dL=T(j); fR,dR=T(j+1)
        for c,g in fL: M[row,c]+=g(Rj)
        for c,g in fR: M[row,c]-=g(Rj)
        row+=1
        for c,g in dL: M[row,c]+=-eps[j]*g(Rj)
        for c,g in dR: M[row,c]-=-eps[j+1]*g(Rj)
        if j==0: b[row]=-Ps/EPS0
        row+=1
    x=np.linalg.solve(M,b)
    def E(r):
        i=int(np.searchsorted(radii,r)); _,dd=T(i)
        return abs(sum(x[c]*g(r) for c,g in dd))
    return E
def run(s,c_nm,t_shell=0.0,use_corr=True,use_sat=True,N=160,out=40.0):
    er,nn=SOLV[s]; Ec=EC[s]; einf=nn*nn; xis=c_nm*XIREL[s]*1e-9
    Rs=RC+t_shell*1e-9
    edges=Rs+np.linspace(0,out,N+1)*1e-9
    mids=0.5*(edges[:-1]+edges[1:]); dloc=mids-Rs
    corr=(1-np.exp(-dloc/xis)) if use_corr else 1.0
    eps_sh=np.full(N,er)
    pre=[RC,Rs] if t_shell>0 else [RC]
    epre=[6.0,3.9] if t_shell>0 else [6.0]
    radii=np.array(pre+list(edges[1:-1])+[edges[-1]])
    for _ in range(80):
        E=solve_ml(radii,np.array(epre+list(eps_sh)+[er]))
        Em=np.array([E(r) for r in mids])
        L=Lfac(Em/Ec) if use_sat else 1.0
        new=einf+(er-einf)*corr*L
        if np.max(np.abs(new-eps_sh)/np.maximum(eps_sh,1e-9))<1e-4: eps_sh=new; break
        eps_sh=eps_sh*0.6+new*0.4
    E=solve_ml(radii,np.array(epre+list(eps_sh)+[er]))
    return E(Rs+0.5e-9), eps_sh, (mids-Rs)*1e9
def bulkE(s,t_shell=0.0):
    er,_=SOLV[s]; Rs=RC+t_shell*1e-9
    if t_shell>0: return solve_ml(np.array([RC,Rs]),np.array([6.0,3.9,er]))(Rs+0.5e-9)
    return solve_ml(np.array([RC]),np.array([6.0,er]))(Rs+0.5e-9)
E0={s:bulkE(s) for s in SOLV}
def loss(c_nm):
    L=0
    for s in SOLV:
        enh=run(s,c_nm)[0]/E0[s]*SF[s]
        L+=(np.log(enh/MEAS_ENH[s]))**2
    return L
cs=np.linspace(2,30,15)
ls=[loss(c) for c in cs]
c0=cs[int(np.argmin(ls))]
for c in np.linspace(max(2,c0-2),c0+2,9):
    l=loss(c)
    if l<min(ls): c0=c; ls=[l]
print(f"Fitted global scale c = {c0:.1f} nm  ->  xi: "+
      ", ".join(f"{s} {c0*XIREL[s]:.1f}" for s in SOLV))
res={"c":float(c0),"xi":{s:float(c0*XIREL[s]) for s in SOLV}}
print(f"\n{'solvent':>7}{'model enh. (corrected)':>14}{'measured':>8}{'ratio':>7}")
for s in SOLV:
    enh=run(s,c0)[0]/E0[s]*SF[s]
    res[s]={"enh_model":float(enh),"enh_meas":MEAS_ENH[s]}
    print(f"{s:>7}{enh:>14.2f}{MEAS_ENH[s]:>8.2f}{enh/MEAS_ENH[s]:>7.2f}")
print(f"\n{'d':>4}{'full model':>9}{'saturation only':>8}{'measured':>7}")
D=np.load("ef_data.npy",allow_pickle=True).item()
pred={"full":[],"sat":[],"meas":[]}
for i,d in enumerate((0,5,10,15)):
    Eh=run("HexOH",c0,d)[0]; Em=run("MeOH",c0,d)[0]
    Eh_s=run("HexOH",1e9,d,use_corr=False)[0]; Em_s=run("MeOH",1e9,d,use_corr=False)[0]
    mr=D["HexOH"]["EF"][i]/D["MeOH"]["EF"][i]
    full=Eh/Em*SF["HexOH"]/SF["MeOH"]; sat=Eh_s/Em_s*SF["HexOH"]/SF["MeOH"]
    pred["full"].append(float(full)); pred["sat"].append(float(sat)); pred["meas"].append(float(mr))
    print(f"{d:>4}{full:>9.2f}{sat:>8.2f}{mr:>7.2f}")
res["contrast"]=pred
json.dump(res,open("corr_model.json","w"),indent=1)
