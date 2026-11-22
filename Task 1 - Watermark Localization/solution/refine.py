# local refinement of a rectangle proposal by crop decoding (patch type, exact rectangles)
from common import *
import torch.nn.functional as F
RES={"trustmark":245,"pixelseal":256,"stegastamp":400}
class Scorer:
    def __init__(self, meth):
        self.meth=meth; self.w=wm(meth); self.n=0; self.cache={}
    def set(self, img, msg):
        self.x=to_tensor(img); self.s=torch.tensor(np.asarray(msg)*2.0-1); self.cache={}
    def score(self, boxes):
        """boxes: list of (y0,y1,x0,x1) ints -> list of (obj, acc)"""
        todo=[b for b in dict.fromkeys(boxes) if b not in self.cache]
        r=RES[self.meth]
        for i in range(0,len(todo),32):
            bb=todo[i:i+32]
            if self.meth=="stegastamp":
                cr=torch.cat([F.interpolate(self.x[:,:,a:b,c:d],size=(r,r),mode="bicubic",align_corners=False,antialias=True).clamp(0,1) for a,b,c,d in bb])
            else:
                cr=torch.cat([F.interpolate(self.x[:,:,a:b,c:d],size=(r,r),mode="bilinear",align_corners=False,antialias=True) for a,b,c,d in bb])
            z=self.w.logits(cr)*self.s[None]; self.n+=len(bb)
            for b,zz in zip(bb,z): self.cache[b]=(float(torch.tanh(zz/2).mean()), float((zz>0).float().mean()))
        return [self.cache[b] for b in boxes]
def clipbox(b,minsz=6):
    y0,y1,x0,x1=b; y0=max(0,min(y0,512-minsz)); x0=max(0,min(x0,512-minsz)); y1=min(512,max(y1,y0+minsz)); x1=min(512,max(x1,x0+minsz))
    return (int(y0),int(y1),int(x0),int(x1))
def refine(sc, box, steps=(16,8,4,2,1), iters=3):
    """coordinate descent on the 4 edges + translations, coarse to fine; returns (box, obj, acc)"""
    best=clipbox(box); bo,ba=sc.score([best])[0]
    for st in steps:
        h,w_=best[1]-best[0],best[3]-best[2]
        if st>max(2,0.35*min(h,w_)): continue
        for _ in range(iters):
            y0,y1,x0,x1=best; cand=[]
            for d in (-st,st):
                cand+=[(y0+d,y1,x0,x1),(y0,y1+d,x0,x1),(y0,y1,x0+d,x1),(y0,y1,x0,x1+d),(y0+d,y1+d,x0,x1),(y0,y1,x0+d,x1+d)]
            cand+=[(y0-st,y1+st,x0-st,x1+st),(y0+st,y1-st,x0+st,x1-st),(y0-st,y1+st,x0,x1),(y0+st,y1-st,x0,x1),(y0,y1,x0-st,x1+st),(y0,y1,x0+st,x1-st)]
            cand=[clipbox(c) for c in cand]; r=sc.score(cand); j=int(np.argmax([o for o,_ in r]))
            if r[j][0]>bo+1e-6: best,bo,ba=cand[j],r[j][0],r[j][1]
            else: break
    return best,bo,ba
def box_mask128(b):
    m=np.zeros((512,512),np.uint8); m[b[0]:b[1],b[2]:b[3]]=1; return downsample_mask(m)
if __name__=="__main__":
    import sys, time
    meth=sys.argv[1]; jit=float(sys.argv[2]) if len(sys.argv)>2 else .08
    sc=Scorer(meth); rng=np.random.default_rng(0); rows=[]; t=time.time()
    for it in ds("train",meth):
        if it["full_watermark"]: continue
        m=it["mask"]>0; ys,xs=np.where(m); g=(ys.min(),ys.max()+1,xs.min(),xs.max()+1)
        if m[g[0]:g[1],g[2]:g[3]].mean()<=.98: continue
        h,w_=g[1]-g[0],g[3]-g[2]
        # start = GT box seen through a 4-px grid, each edge moved by up to jit*size (+-4 px at least)
        st=[int(round((v+rng.uniform(-1,1)*max(4,jit*s))/4)*4) for v,s in zip(g,(h,h,w_,w_))]
        st=clipbox((st[0],st[1],st[2],st[3]))
        sc.set(it["image"],it["message"]); a0=sc.score([st])[0][1]
        b,bo,ba=refine(sc,st)
        rows.append((miou_image(box_mask128(st),it["mask128"]),miou_image(box_mask128(b),it["mask128"]),a0,ba,max(abs(np.array(b)-np.array(g))),h*w_))
    R=np.array(rows); print(meth,"n",len(R),"jit",jit,"crops/img %.0f  %.1fs/img"%(sc.n/len(R),(time.time()-t)/len(R)))
    print("  start mIoU %.3f acc %.3f -> refined mIoU %.3f acc %.3f"%(R[:,0].mean(),R[:,2].mean(),R[:,1].mean(),R[:,3].mean()))
    for th in (.75,.85,.9,.95):
        k=R[:,3]>=th; print("  acc>=%.2f: n=%d  refined mIoU %.4f (start %.4f)  max edge err median %.1f px, frac exact(<=1px) %.2f, min mIoU %.3f"%(th,k.sum(),R[k,1].mean(),R[k,0].mean(),np.median(R[k,4]),(R[k,4]<=1).mean(),R[k,1].min() if k.any() else 0))
    k=R[:,3]<.9; print("  acc<.9: n=%d refined mIoU %.3f start %.3f"%(k.sum(),R[k,1].mean(),R[k,0].mean()))
