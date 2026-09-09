# Supplementary Software for "Solvent-Reconstructed Electric Fields at
# Solid-Liquid Interfaces" (Cho et al.). MIT License; see LICENSE.
"""Caches the geometric matrix across permittivity updates, so that the
self-consistent scan does not reassemble it at every iteration."""
import warnings,sys; warnings.filterwarnings("ignore")
import numpy as np
from ferroelectric_bem import build_geometry
from multilayer_bem import build_stack, assemble
A_NM=56.7; OFF=np.array([1.5,3.,5.,8.,12.,20.])/A_NM
shape=sys.argv[1]
g=build_geometry("sphere",subdiv=2,R=1.0) if shape=="sphere" \
  else build_geometry("rounded_cube",subdiv=2,a=1.0,p=5.5)
st=build_stack(g,list(OFF))
asm=assemble(st)
np.savez(f"mlM_{shape}.npz",M=asm["M"],C=asm["C"],N=asm["N"],Ar=asm["Ar"],
         sizes=np.array(asm["sizes"]))
np.save(f"mlG_{shape}.npy",{"tri0":np.vstack([x["tri"][0] for x in st]),
 "tri1":np.vstack([x["tri"][1] for x in st]),"tri2":np.vstack([x["tri"][2] for x in st])},
 allow_pickle=True)
print(shape,"assembled:",sum(asm["sizes"]),"panels")
