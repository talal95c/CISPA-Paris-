# apply2.py base.pt out.pt : apply.py rule (lr<=-0.6, shrink 3) + pixelseal boxes with lr<=-0.4 whose crop decodes the known message (acc>=0.9)
from common import *
import sys, glob, pickle
base,out=sys.argv[1],sys.argv[2]
d=torch.load(base,weights_only=False); names=list(d["image_names"]); idx={n:i for i,n in enumerate(names)}; M=torch.as_tensor(d["masks"]).clone().bool()
R={}
for f in sorted(glob.glob("rl/test_*.pkl")):
    for n,r in pickle.load(open(f,"rb")).items():
        if n not in R or r["lr"]<R[n]["lr"]: R[n]=dict(box=r["box"],lr=r["lr"],src=f)
for f in sorted(glob.glob("rl/kid_*.json")):
    for n,r in json.load(open(f)).items():
        if n not in R or r["lr"]<R[n]["lr"]: R[n]=dict(box=r["box"],lr=r["lr"],src=f)
w=wm("pixelseal"); D={it["image_name"]:it for it in ds("test","pixelseal")}
def acc(n,b):
    it=D[n]; y0,y1,x0,x1=b; C=np.ascontiguousarray(np.asarray(it["image"])[y0:y1,x0:x1])
    return float((w.decode(to_tensor(C))[0].numpy()==(np.asarray(it["message"]).astype(float)>0.5)).mean())
k=0
for n,r in R.items():
    ok=r["lr"]<=-0.6
    if not ok and r["lr"]<=-0.4 and n in D:
        a=acc(n,r["box"]); ok=a>=0.9; print("pixelseal mid-lr",n,r["box"],"lr %.2f acc %.2f"%(r["lr"],a),"ACCEPT" if ok else "reject")
    if not ok: continue
    y0,y1,x0,x1=r["box"]; m=np.zeros((512,512),np.uint8); m[y0+3:y1-3,x0+3:x1-3]=1
    m=torch.as_tensor(np.asarray(downsample_mask(m))>0); o=M[idx[n]]
    if (m!=o).any(): print("changed",n,"iou %.2f lr %.2f"%((m&o).sum().item()/max(1,(m|o).sum().item()),r["lr"]),r["src"])
    M[idx[n]]=m; k+=1
print("applied",k)
torch.save({"image_names":names,"masks":M},out); print(out,M.shape,M.dtype,os.path.getsize(out))
