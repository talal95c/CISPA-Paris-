# sharpness of the re-embedding energy ratio as a rectangle objective (exact-rectangle patch images, train)
from common import *
import sys, time
meth=sys.argv[1]; N=int(sys.argv[2]); w=wm(meth); n=0; t=time.time()
def lr(X,m,b):
    y0,y1,x0,x1=b; C=np.ascontiguousarray(X[y0:y1,x0:x1]); Cf=C.astype(np.float32)
    e1=((w.embed(C,m).astype(np.float32)-Cf)**2).sum(); e0=((w.embed(C,1-m).astype(np.float32)-Cf)**2).sum()
    return float(np.log((e1+1)/(e0+1)))
for it in ds("train",meth):
    if it["full_watermark"] or n>=N: continue
    M=it["mask"]>0; ys,xs=np.where(M)
    if len(ys)==0: continue
    y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
    if M.sum()<.97*(y1-y0)*(x1-x0) or min(y1-y0,x1-x0)<48: continue
    n+=1; X=it["image"]; m=it["message"]; g=lr(X,m,(y0,y1,x0,x1)); row=[]
    for d in (-16,-8,-4,-2,2,4,8,16):
        v=[lr(X,m,b) for b in [(max(0,y0+d),y1,x0,x1),(y0,y1,max(0,x0+d),x1),(y0,min(512,y1+d),x0,x1),(y0,y1,x0,min(512,x1+d))]]
        row.append("%+d:%.2f"%(d,np.mean(v)))
    print("box %dx%d GT lr %.2f | one edge moved: "%(y1-y0,x1-x0,g)+" ".join(row),"%.1fs"%((time.time()-t)/n),flush=True)
