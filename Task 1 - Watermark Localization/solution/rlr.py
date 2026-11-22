# rlr.py meth mode(train|test) N out [SH=i/k] : rectangle refinement with the re-embedding energy ratio (sharp at the exact frame)
from common import *
import sys, time, pickle
meth,mode,N,out=sys.argv[1],sys.argv[2],int(sys.argv[3]),sys.argv[4]
EARLY=float(os.environ.get('EARLY','-0.12'))
SH=[int(v) for v in os.environ.get("SH","0/1").split("/")]; w=wm(meth); rng=np.random.default_rng(0)
class Obj:
    def __init__(s,X,m): s.X=X; s.m=m; s.c={}
    def __call__(s,b):
        if b not in s.c:
            y0,y1,x0,x1=b; C=np.ascontiguousarray(s.X[y0:y1,x0:x1]); Cf=C.astype(np.float32)
            e1=((w.embed(C,s.m).astype(np.float32)-Cf)**2).sum(); e0=((w.embed(C,1-s.m).astype(np.float32)-Cf)**2).sum()
            s.c[b]=float(np.log((e1+1)/(e0+1)))
        return s.c[b]
def clip(b,mn=24):
    y0,y1,x0,x1=b; y0=max(0,min(y0,512-mn)); x0=max(0,min(x0,512-mn)); y1=min(512,max(y1,y0+mn)); x1=min(512,max(x1,x0+mn)); return (int(y0),int(y1),int(x0),int(x1))
def refine(f,b,steps=(8,4,2,1),iters=4):
    b=clip(b); v=f(b)
    for st in steps:
        if st<steps[0] and v>EARLY: break
        for _ in range(iters):
            y0,y1,x0,x1=b; c=[]
            for d in (-st,st):
                c+=[(y0+d,y1,x0,x1),(y0,y1+d,x0,x1),(y0,y1,x0+d,x1),(y0,y1,x0,x1+d),(y0+d,y1+d,x0,x1),(y0,y1,x0+d,x1+d),(y0-d,y1+d,x0-d,x1+d)]
            c=[clip(q) for q in c]; r=[f(q) for q in c]; j=int(np.argmin(r))
            if r[j]<v-1e-4: b,v=c[j],r[j]
            else: break
    return b,v
def biou(a,b):
    ih=max(0,min(a[1],b[1])-max(a[0],b[0])); iw=max(0,min(a[3],b[3])-max(a[2],b[2])); i=ih*iw
    return i/((a[1]-a[0])*(a[3]-a[2])+(b[1]-b[0])*(b[3]-b[2])-i)
jobs=[]
if mode=="train":
    for it in ds("train",meth):
        if it["full_watermark"]: continue
        M=it["mask"]>0; ys,xs=np.where(M)
        if len(ys)==0: continue
        g=(int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1)
        if M.sum()<.97*(g[1]-g[0])*(g[3]-g[2]) or min(g[1]-g[0],g[3]-g[2])<40: continue
        s=clip(tuple(int(v) for v in np.array(g)+rng.integers(-10,11,4)))
        jobs.append((it,[s],g))
elif mode=="trainobj":   # non-rectangle patches: does lr falsely lock?
    for it in ds("train",meth):
        if it["full_watermark"]: continue
        M=it["mask"]>0; ys,xs=np.where(M)
        if len(ys)==0: continue
        g=(int(ys.min()),int(ys.max())+1,int(xs.min()),int(xs.max())+1); fill=M.sum()/((g[1]-g[0])*(g[3]-g[2]))
        if fill>=.9 or fill<.5 or min(g[1]-g[0],g[3]-g[2])<40: continue
        jobs.append((it,[clip((g[0]-3,g[1]+3,g[2]-3,g[3]+3))],g))
else:
    S=pickle.load(open(sys.argv[5],"rb")); S={k:[(b[0]-3,b[1]+3,b[2]-3,b[3]+3) for b in v] for k,v in S.items()}   # name -> list of start boxes
    for it in ds("test",meth):
        if it["image_name"] in S: jobs.append((it,S[it["image_name"]],None))
R={}; t=time.time(); n=0
jobs=list(enumerate(jobs))
if os.environ.get('REV'): jobs=jobs[::-1]
for j,(it,starts,g) in jobs:
    if j%SH[1]!=SH[0] or n>=N: continue
    n+=1; f=Obj(it["image"],it["message"]); best=None
    for s in starts:
        b,v=refine(f,tuple(int(q) for q in s))
        if best is None or v<best[1]: best=(b,v,f(clip(tuple(int(q) for q in s))))
    R[it["image_name"]]=dict(box=best[0],lr=best[1],lr0=best[2],starts=starts,gt=g)
    print(n,it["image_name"],"box",best[0],"lr %.2f (start %.2f)"%(best[1],best[2]),("IoU start %.3f -> %.3f"%(biou(starts[0],g),biou(best[0],g))) if g else "","evals %d %.0fs/img"%(len(f.c),(time.time()-t)/n),flush=True)
    pickle.dump(R,open(out,"wb"))
