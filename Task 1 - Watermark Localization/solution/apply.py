# apply.py base.pt out.pt [THR=-0.6] : replace masks by lr-refined rectangles (rl/test_*.pkl + rl/kid_*.json), shrunk by 3 px
from common import *
import sys, glob, pickle
base,out=sys.argv[1],sys.argv[2]; THR=float(sys.argv[3]) if len(sys.argv)>3 else -0.6
d=torch.load(base,weights_only=False); names=list(d["image_names"]); idx={n:i for i,n in enumerate(names)}; M=torch.as_tensor(d["masks"]).clone().bool()
R={}
for f in sorted(glob.glob("rl/test_*.pkl")):
    for n,r in pickle.load(open(f,"rb")).items():
        if n not in R or r["lr"]<R[n]["lr"]: R[n]=dict(box=r["box"],lr=r["lr"],src=f)
for f in sorted(glob.glob("rl/kid_*.json")):
    for n,r in json.load(open(f)).items():
        if n not in R or r["lr"]<R[n]["lr"]: R[n]=dict(box=r["box"],lr=r["lr"],src=f)
k=0; ious=[]; new=0
for n,r in R.items():
    if r["lr"]>THR: continue
    y0,y1,x0,x1=r["box"]; m=np.zeros((512,512),np.uint8); m[y0+3:y1-3,x0+3:x1-3]=1
    m=torch.as_tensor(np.asarray(downsample_mask(m))>0); o=M[idx[n]]
    i=(m&o).sum().item()/max(1,(m|o).sum().item()); ious.append(i); new+=i<.3; M[idx[n]]=m; k+=1
print("refined rects in pool",len(R),"applied",k,"IoU with previous mask: mean %.3f, q10 %.3f, <0.3: %d"%(np.mean(ious),np.quantile(ious,.1),new))
torch.save({"image_names":names,"masks":M},out); print(out,M.shape,M.dtype,os.path.getsize(out))
