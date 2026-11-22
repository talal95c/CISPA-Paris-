# sharpness of crop-decoding objectives around the GT rectangle edges (1-D perturbation of each edge / of both size params)
from common import *
import sys, torch.nn.functional as F
meth=sys.argv[1]; N=int(sys.argv[2]); from refine import RES
w=wm(meth); D=(-8,-4,-2,-1,0,1,2,4,8); C={k:[] for k in ("raw","tanh","p10","acc")}; n=0; hit={k:[] for k in C}
for it in ds("train",meth):
    if it["full_watermark"] or n>=N: continue
    m=it["mask"]>0; ys,xs=np.where(m); g=[ys.min(),ys.max()+1,xs.min(),xs.max()+1]
    if m[g[0]:g[1],g[2]:g[3]].mean()<=.98 or min(g[1]-g[0],g[3]-g[2])<40: continue
    x=to_tensor(it["image"]); s=torch.tensor(it["message"]*2.0-1); r=RES[meth]
    def sc(b):
        a,bb,c,d=[int(min(512,max(0,v))) for v in b]
        cr=F.interpolate(x[:,:,a:bb,c:d],size=(r,r),mode="bicubic" if meth=="stegastamp" else "bilinear",align_corners=False,antialias=True).clamp(0,1)
        z=w.logits(cr)[0]*s
        return dict(raw=float(z.mean()),tanh=float(torch.tanh(z/2).mean()),p10=float(torch.quantile(z,.1)),acc=float((z>0).float().mean()))
    if sc(g)["acc"]<.9: continue
    n+=1
    for e in range(4):
        if (e in (0,2) and g[e]<=8) or (e in (1,3) and g[e]>=504): continue
        cur=[sc([v+(d if j==e else 0) for j,v in enumerate(g)]) for d in D]
        for k in C:
            v=np.array([c[k] for c in cur]); C[k].append(v); hit[k].append(abs(D[int(np.argmax(v))]))
print(meth,"images",n,"edges",len(C["raw"]),"  d =",D)
for k in C:
    v=np.array(C[k]); print("  %-5s mean curve"%k,np.round(v.mean(0),3)," argmax |d|<=1: %.2f  <=2: %.2f"%((np.array(hit[k])<=1).mean(),(np.array(hit[k])<=2).mean()))
