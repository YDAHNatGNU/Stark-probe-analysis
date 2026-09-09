# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Self-consistent graded-permittivity model for a spherical particle. The solvent is
divided into thin shells; the permittivity of each shell is set by Langevin
saturation at the local field, and the field is re-solved until convergence. Unlike
a static interfacial layer, the permittivity profile is determined by the solution
itself."""
import numpy as np, json
EPS0, Ps = 8.8541878128e-12, 0.26
RC = 56.7e-9
SOLV={"MeOH":(33.7,1.3288),"EtOH":(24.5,1.3611),"PrOH":(20.1,1.3856),"HexOH":(13.0,1.4178)}
B=np.load("booth_results.npy",allow_pickle=True).item()
EC={s:B["PAR"][s]["E_c"] for s in SOLV}
def Lfac(x):
    x=np.maximum(x,1e-6); return (3/x)*(1/np.tanh(x)-1/x)
def solve_multilayer(radii, eps):
    """l = 1 multilayer solution. radii (m) are the interfaces in ascending order and
    eps holds the permittivity of each region (len+1). Returns E(r)."""
    K=len(radii); n=2*K
    M=np.zeros((n,n)); b=np.zeros(n)
    def terms(i):
        if i==0: return [(0,lambda r:r)],[(0,lambda r:1.0)]
        if i==K: return [(n-1,lambda r:r**-2)],[(n-1,lambda r:-2*r**-3)]
        a=1+2*(i-1)
        return ([(a,lambda r:r),(a+1,lambda r:r**-2)],
                [(a,lambda r:1.0),(a+1,lambda r:-2*r**-3)])
    row=0
    for j,Rj in enumerate(radii):
        fL,dL=terms(j); fR,dR=terms(j+1)
        for c,g in fL: M[row,c]+=g(Rj)
        for c,g in fR: M[row,c]-=g(Rj)
        row+=1
        for c,g in dL: M[row,c]+=-eps[j]*g(Rj)
        for c,g in dR: M[row,c]-=-eps[j+1]*g(Rj)
        if j==0: b[row]=-Ps/EPS0
        row+=1
    x=np.linalg.solve(M,b)
    def E_axis(r):
        i=int(np.searchsorted(radii,r)); _,dd=terms(i)
        return abs(sum(x[c]*g(r) for c,g in dd))
    return E_axis
def run(s, t_shell_nm, N=200, out_nm=40.0, itmax=60, mix=0.4):
    er,nn=SOLV[s]; Ec=EC[s]; e_inf=nn*nn
    Rs=RC+t_shell_nm*1e-9
    edges=Rs+np.linspace(0,out_nm,N+1)*1e-9
    mids=0.5*(edges[:-1]+edges[1:])
    eps_sh=np.full(N,er)
    for it in range(itmax):
        radii=[RC]+([Rs] if t_shell_nm>0 else [])+list(edges[1:-1])+[edges[-1]]
        eps=[6.0]+([3.9] if t_shell_nm>0 else [])+list(eps_sh)+[er]
        Ef=solve_multilayer(np.array(radii),np.array(eps))
        Emid=np.array([Ef(r) for r in mids])
        new=e_inf+(er-e_inf)*Lfac(Emid/Ec)
        if np.max(np.abs(new-eps_sh)/eps_sh)<1e-4:
            eps_sh=new; break
        eps_sh=eps_sh*(1-mix)+new*mix
    rp=Rs+0.5e-9
    return Ef(rp), Ef, eps_sh, mids
D=np.load("ef_data.npy",allow_pickle=True).item()
print("Self-consistent saturation model (sphere, graded eps(E))")
print(f"{'d(nm)':>7}{'model contrast':>10}{'measured contrast':>10}{'  MeOH enh.':>10}{'HexOH enh.':>10}")
res={}
for i,d in enumerate((0,5,10,15)):
    Eh,_,_,_=run("HexOH",d); Em,_,_,_=run("MeOH",d)
    er,nn=SOLV["MeOH"]
    Rs=RC+d*1e-9
    rad=[RC]+([Rs] if d>0 else []); ep=[6.0]+([3.9] if d>0 else [])+[er]
    Em0=solve_multilayer(np.array(rad),np.array(ep))(Rs+0.5e-9)
    er,nn=SOLV["HexOH"]; ep=[6.0]+([3.9] if d>0 else [])+[er]
    Eh0=solve_multilayer(np.array(rad),np.array(ep))(Rs+0.5e-9)
    mr=D["HexOH"]["EF"][i]/D["MeOH"]["EF"][i]
    res[d]={"model_ratio":Eh/Em,"meas_ratio":float(mr),
            "enh_MeOH":Em/Em0,"enh_HexOH":Eh/Eh0}
    print(f"{d:>7}{Eh/Em:>10.2f}{mr:>10.2f}{Em/Em0:>10.2f}{Eh/Eh0:>10.2f}")
json.dump(res,open("selfcons_scan.json","w"),indent=1)
_,_,pM,mM=run("MeOH",0); _,_,pH,mH=run("HexOH",0)
np.save("selfcons_prof.npy",{"MeOH":(mM,pM),"HexOH":(mH,pH)},allow_pickle=True)
er,nn=SOLV["MeOH"]
print(f"\nSurface permittivity at d = 0: MeOH {pM[0]:.1f} (bulk {er}), ", end="")
er,nn=SOLV["HexOH"]
print(f"HexOH {pH[0]:.1f} (bulk {er})")
i90=np.argmax(pM>0.9*33.7); j90=np.argmax(pH>0.9*13.0)
print(f"Distance for 90% recovery: MeOH {(mM[i90]-RC)*1e9:.1f} nm, HexOH {(mH[j90]-RC)*1e9:.1f} nm")
