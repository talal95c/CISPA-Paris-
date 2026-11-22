# fill.py in.pt out.pt : a GT mask is never all-zero / all-one (train: 0/510) -> where the base is degenerate, fall back to the second model
from common import *
import sys
b=torch.load(sys.argv[1],weights_only=False); names=list(b["image_names"]); idx={n:i for i,n in enumerate(names)}; M=torch.as_tensor(b["masks"]).clone().bool()
A=np.load("cache/bitacc.npz",allow_pickle=True); acc=dict(zip([str(n) for n in A["names"]],np.asarray(A["vals"])[:,0]))
k=np.load("../kids/pixelseal_patch_v2.npz",allow_pickle=True); n1=0
for j,n in enumerate(k["names"]):
    n=str(n); m=k["masks"][j]>0
    if acc[n]<.80 and M[idx[n]].sum()==0 and m.sum()>0: M[idx[n]]=torch.as_tensor(m); n1+=1
k=np.load("../kids/trustmark_hole_v2.npz",allow_pickle=True); n2=0
for j,n in enumerate(k["names"]):
    n=str(n); m=k["masks"][j]>0
    if acc[n]>=.78 and bool(M[idx[n]].all()) and not m.all(): M[idx[n]]=torch.as_tensor(m); n2+=1
print("filled pixelseal empty patches",n1,"trustmark full holes",n2)
torch.save({"image_names":names,"masks":M},sys.argv[2]); print(sys.argv[2],M.shape,M.dtype,os.path.getsize(sys.argv[2]))
