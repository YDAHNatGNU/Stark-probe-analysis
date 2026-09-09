# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Sphere versus measured shape: self-consistent against linear solutions, evaluated
as a polar-cap average, with the resulting shape correction factor."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, json
from ferroelectric_bem import build_geometry
from core_shell_bem import offset_mesh
from accurate_bem import tri_int_vec
from scipy.linalg import solve as lasolve
EPS0,Ps,A_NM=8.8541878128e-12,0.26,56.7
SOLV={"MeOH":(33.7,1.3288),"HexOH":(13.0,1.4178)}
B=np.load("booth_results.npy",allow_pickle=True).item()
EC={s:B["PAR"][s]["E_c"] for s in SOLV}
MID=np.array([0.75,2.25,4.,6.5,10.,16.])
def Lfac(x): x=np.maximum(x,1e-6); return (3/x)*(1/np.tanh(x)-1/x)
def load(shape):
    z=np.load(f"mlM_{shape}.npz"); G=np.load(f"mlG_{shape}.npy",allow_pickle=True).item()
    g=build_geometry("sphere",subdiv=2,R=1.0) if shape=="sphere" \
      else build_geometry("rounded_cube",subdiv=2,a=1.0,p=5.5)
    return dict(M=z["M"],N=z["N"],sizes=z["sizes"],G=G,gc=g)
def field_pts(D,sig,pts):
    k=1.0/(4.0*np.pi*EPS0); G=D["G"]
    E=np.empty((len(pts),3))
    for m,P in enumerate(pts):
        J=tri_int_vec(P,G["tri0"],G["tri1"],G["tri2"])
        E[m]=k*(sig[:,None]*J).sum(axis=0)
    return E
def cap_pts(D,off_nm):
    """Centroids and areas of the polar cap (within 25 deg of the z axis)
    on the surface offset by 0.5 nm."""
    gp=offset_mesh(D["gc"],off_nm/A_NM)
    C=gp["centroids"]; A=gp["areas"]
    ct=C[:,2]/np.linalg.norm(C,axis=1)
    m=ct>np.cos(np.radians(25))
    return C[m],A[m]
def solve(D,eps_reg,):
    M=D["M"];N=D["N"];sz=D["sizes"];n=len(N)
    lam=np.empty(n);b=np.zeros(n);i0=0
    for j,s_ in enumerate(sz):
        ein,eout=eps_reg[j],eps_reg[j+1]
        lam[i0:i0+s_]=(eout-ein)/(eout+ein)
        if j==0: b[i0:i0+s_]=(2.0/(ein+eout))*Ps*N[i0:i0+s_,2]
        i0+=s_
    A=(2.0*lam)[:,None]*M.copy()
    A[np.diag_indices(n)]+=1.0
    return lasolve(A,b,overwrite_a=True,check_finite=False)
def probe_val(D,sig,cp,ca):
    E=np.linalg.norm(field_pts(D,sig,cp),axis=1)
    return float(np.sqrt(np.sum(ca*E**2)/np.sum(ca)))
res={}
for shape in ("sphere","cube"):
    D=load(shape); cp,ca=cap_pts(D,0.5)
    mids=np.array([[0,0,0,],])  # placeholder
    zsurf=1.0
    mpts=np.array([[0,0,zsurf+m/A_NM] for m in MID])
    res[shape]={}
    for s in SOLV:
        er,nn=SOLV[s]; Ec=EC[s]; einf=nn*nn
        eps_l=np.full(6,er)
        for it in range(40):
            sig=solve(D,np.concatenate([[6.0],eps_l,[er]]))
            Em=np.linalg.norm(field_pts(D,sig,mpts),axis=1)
            new=einf+(er-einf)*Lfac(Em/Ec)
            if np.max(np.abs(new-eps_l)/eps_l)<1e-4: eps_l=new; break
            eps_l=0.6*eps_l+0.4*new
        sig=solve(D,np.concatenate([[6.0],eps_l,[er]]))
        Esc=probe_val(D,sig,cp,ca)
        sig0=solve(D,np.concatenate([[6.0],np.full(6,er),[er]]))
        Elin=probe_val(D,sig0,cp,ca)
        res[shape][s]={"sc":Esc,"lin":Elin,"enh":Esc/Elin,"eps":eps_l.tolist()}
        print(f"{shape:>7} {s:>6}: self-consistent {Esc/1e8:6.2f}  linear {Elin/1e8:6.2f}  "
              f"enhancement {Esc/Elin:.3f}")
for shape in ("sphere","cube"):
    r=res[shape]
    print(f"{shape:>7} contrast(self-consistent) {r['HexOH']['sc']/r['MeOH']['sc']:.3f}  "
          f"(linear) {r['HexOH']['lin']/r['MeOH']['lin']:.3f}")
json.dump(res,open("shape_selfcons.json","w"),indent=1)
