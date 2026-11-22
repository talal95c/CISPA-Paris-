# build.py BASE.pt OUT.pt meth:type:kid.npz[:mode] ...   mode: swap (default) | rect:TH (only where kid rect_acc>=TH) | and | or
from common import *
import sys
B=torch.load(sys.argv[1],weights_only=True); names=list(B["image_names"]); M=B["masks"].clone().bool(); idx={n:i for i,n in enumerate(names)}
z=np.load("cache/bitacc.npz"); ACC={str(n):v[0] for n,v in zip(z["names"],z["vals"])}
THR={"stegastamp":.75,"trustmark":.78,"pixelseal":.80}
METH={it["image_name"]:it["method"] for it in ds("test")}
for spec in sys.argv[3:]:
    p=spec.split(":"); meth,typ,f=p[:3]; mode=p[3] if len(p)>3 else "swap"; th=float(p[4]) if len(p)>4 else .9
    k=np.load(f,allow_pickle=True); n=0; ch=[]
    for j,nm in enumerate(k["names"]):
        nm=str(nm)
        if METH[nm]!=meth or (ACC[nm]>=THR[meth])!=(typ=="hole"): continue
        m=torch.tensor(k["masks"][j]>0)
        if mode=="rect":
            if "rect_acc" not in k.files or k["rect_acc"][j]<th: continue
        elif mode=="and": m=m&M[idx[nm]]
        elif mode=="or": m=m|M[idx[nm]]
        ch.append((m!=M[idx[nm]]).float().mean().item()); M[idx[nm]]=m; n+=1
    print(spec,"-> replaced",n,"mean changed frac %.3f"%(np.mean(ch) if ch else 0))
assert M.shape==(1020,128,128) and M.dtype==torch.bool and len(set(names))==1020
torch.save({"image_names":names,"masks":M},sys.argv[2]); print(sys.argv[2],os.path.getsize(sys.argv[2]))
