# is the clean content JPEG-quantised on the 8x8 grid (then marked blocks break it)? per-block DCT, test periodicity of coefficients in clean vs marked blocks
from common import *
import cv2, sys
from scipy.fft import dctn
d=ds("train"); rng=np.random.default_rng(0)
def blocks(Y,oy=0,ox=0):
    H=(Y.shape[0]-oy)//8*8; W=(Y.shape[1]-ox)//8*8
    B=Y[oy:oy+H,ox:ox+W].reshape(H//8,8,W//8,8).transpose(0,2,1,3)
    return dctn(B,axes=(2,3),norm="ortho")
for i in rng.permutation(len(d))[:8]:
    it=d[int(i)]; X=it["image"].astype(np.float32)
    Y=0.299*X[...,0]+0.587*X[...,1]+0.114*X[...,2]-128
    m=(it["mask"]>0).reshape(64,8,64,8).mean((1,3))
    out=[]
    for (oy,ox) in ((0,0),(4,4)):
        D=blocks(Y,oy,ox)
        if oy: mm=m[:D.shape[0],:D.shape[1]]
        else: mm=m
        res=[]
        for q in range(1,13):
            # periodicity score of coefficient (0,1),(1,0),(1,1) w.r.t. step q: mean cos(2 pi c / q)
            c=D[:,:,[0,1,1],[1,0,1]]
            s=np.cos(2*np.pi*c/q).mean(2)
            res.append((s[mm<.01].mean() if (mm<.01).any() else np.nan, s[mm>.99].mean() if (mm>.99).any() else np.nan))
        out.append(res)
    print(it["method"],"full",it["full_watermark"],"frac %.2f"%m.mean())
    for (oy,ox),res in zip(((0,0),(4,4)),out):
        print("   grid",oy,"clean:"," ".join("%.2f"%a for a,b in res)); print("          marked:"," ".join("%.2f"%b for a,b in res))
