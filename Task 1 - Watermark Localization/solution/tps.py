# tps.py i : train check of the lr refinement for pixelseal, start = child rect (decoded) expanded by 3; compares found box with GT bbox
from common import *
import sys, pickle
k=np.load("../kids/pixelseal_patch_v2.npz",allow_pickle=True); w=wm("pixelseal")
G={it["image_name"]:it for it in ds("train","pixelseal")}
J=[j for j in range(85) if k["train_rect_acc"][j]>=.9]; j=J[int(sys.argv[1])]; n=str(k["train_names"][j]); it=G[n]
X=np.asarray(it["image"]); msg=np.asarray(it["message"])
def clip(b):
    y0,y1,x0,x1=b; y0=max(0,y0); x0=max(0,x0); y1=min(512,y1); x1=min(512,x1)
    if y1-y0<16: y1=min(512,y0+16); y0=y1-16
    if x1-x0<16: x1=min(512,x0+16); x0=x1-16
    return (int(y0),int(y1),int(x0),int(x1))
c={}
def f(b):
    if b not in c:
        y0,y1,x0,x1=b; C=np.ascontiguousarray(X[y0:y1,x0:x1]); Cf=C.astype(np.float32)
        e1=((w.embed(C,msg).astype(np.float32)-Cf)**2).sum(); e0=((w.embed(C,1-msg).astype(np.float32)-Cf)**2).sum()
        c[b]=float(np.log((e1+1)/(e0+1)))
    return c[b]
def refine(b,steps=(8,4,2,1),iters=4):
    b=clip(b); v=f(b)
    for st in steps:
        for _ in range(iters):
            y0,y1,x0,x1=b; cs=[]
            for d in (-st,st):
                cs+=[(y0+d,y1,x0,x1),(y0,y1+d,x0,x1),(y0,y1,x0+d,x1),(y0,y1,x0,x1+d),(y0+d,y1+d,x0,x1),(y0,y1,x0+d,x1+d),(y0-d,y1+d,x0-d,x1+d)]
            cs=[clip(q) for q in cs]; r=[f(q) for q in cs]; i=int(np.argmin(r))
            if r[i]<v-1e-4: b,v=cs[i],r[i]
            else: break
    return b,v
M=it["mask"]>0; ys,xs=np.where(M); g=(int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1)
s=tuple(int(v) for v in k["train_rect"][j]); s=(s[0]-3,s[1]+3,s[2]-3,s[3]+3)
b,v=refine(s)
print("TPS",n,"gt",g,"fill %.2f"%(M.sum()/((g[1]-g[0])*(g[3]-g[2]))),"start",s,"lr0 %.2f"%f(clip(s)),"found",b,"lr %.2f"%v,"lr@gt+3 %.2f"%f(clip((g[0]-3,g[1]+3,g[2]-3,g[3]+3))),"evals",len(c),flush=True)
