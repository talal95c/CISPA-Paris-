# matched filter: re-embed observed image with the known message, correlate predicted residual with high-pass of image
from common import *
import cv2, sys
from sklearn.metrics import roc_auc_score
meth=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 30
w=wm(meth); d=ds("train",meth)
def hp(x,s): return x-cv2.GaussianBlur(x,(0,0),s)
res={}
for i in range(len(d)):
    it=d[i]
    if len(res.setdefault(it["full_watermark"],[]))>=N: continue
    I=it["image"].astype(np.float32)
    W=w.embed(it["image"], it["message"]).astype(np.float32)
    r=W-I
    out=[]
    for s in (1.0,2.0,4.0):
        a=hp(I,s); b=hp(r,s)
        for win in (4,8,16,32):
            num=cv2.blur((a*b).sum(2),(win,win)); den=cv2.blur((b*b).sum(2),(win,win))+1e-3
            m=(num/den)
            m128=m.reshape(128,4,128,4).mean((1,3))
            g=it["mask128"].ravel()
            if 0<g.sum()<g.size: out.append(roc_auc_score(g,m128.ravel()))
            else: out.append(np.nan)
    res[it["full_watermark"]].append(out+[np.abs(r).mean()])
for k,v in res.items():
    v=np.array(v); print(meth,"full=",k,"resid|r|",v[:,-1].mean().round(2)); print(np.round(np.nanmean(v[:,:-1],0).reshape(3,4),3)); print(np.round(np.nanmedian(v[:,:-1],0).reshape(3,4),3))
