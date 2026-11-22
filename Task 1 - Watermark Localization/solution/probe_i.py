# probe_i.py in.pt out.pt LRT : trustmark hole kid v2 where its frame fit is confident (test_logratio < LRT) and image is hole type
from common import *
import sys
b=torch.load(sys.argv[1],weights_only=False); names=list(b["image_names"]); idx={n:i for i,n in enumerate(names)}; M=torch.as_tensor(b["masks"]).clone().bool(); T=float(sys.argv[3])
A=np.load("cache/bitacc.npz",allow_pickle=True); acc=dict(zip([str(n) for n in A["names"]],np.asarray(A["vals"])[:,0]))
k=np.load("../kids/trustmark_hole_v2.npz",allow_pickle=True); n2=0; ch=[]
for j,n in enumerate(k["names"]):
    n=str(n); m=torch.as_tensor(k["masks"][j]>0)
    if acc[n]>=.78 and k["test_logratio"][j]<T: ch.append((M[idx[n]]!=m).float().mean().item()); M[idx[n]]=m; n2+=1
print("swapped",n2,"mean changed frac %.3f"%np.mean(ch))
torch.save({"image_names":names,"masks":M},sys.argv[2]); print(sys.argv[2],M.shape,M.dtype,os.path.getsize(sys.argv[2]))
