# finer frame search: aspect x axis (centre crop), then offset along the padded axis for the best aspect
from common import *
import cv2, sys
from sklearn.metrics import roc_auc_score
meth=sys.argv[1]; N=int(sys.argv[2]); typ=sys.argv[3] if len(sys.argv)>3 else "hole"; w=wm(meth); d=ds("train",meth); n=0
AS=[1.0,1.1,1.143,1.2,1.25,1.286,1.333,1.4,1.46,1.5,1.6,1.75]
def cue(X,m,a,axis,off=.5):
    T=int(round(512*(a-1))); P0=int(round(T*off)); P1=T-P0
    Xp=X if T==0 else (cv2.copyMakeBorder(X,0,0,P0,P1,cv2.BORDER_REFLECT_101) if axis==1 else cv2.copyMakeBorder(X,P0,P1,0,0,cv2.BORDER_REFLECT_101))
    Xf=Xp.astype(np.float32)
    e1=((w.embed(Xp,m).astype(np.float32)-Xf)**2).sum(2); e0=((w.embed(Xp,1-m).astype(np.float32)-Xf)**2).sum(2)
    if T: e1,e0=(e1[:,P0:P0+512],e0[:,P0:P0+512]) if axis==1 else (e1[P0:P0+512],e0[P0:P0+512])
    return e1,e0
def stat(e1,e0,g):
    L=np.log((cv2.blur(e1,(64,64))+.05)/(cv2.blur(e0,(64,64))+.05)); return np.log(e1.sum()/e0.sum()), roc_auc_score(g,-L.reshape(128,4,128,4).mean((1,3)).ravel())
for it in d:
    if it["full_watermark"]!=(typ=="hole") or n>=N: continue
    n+=1; X=it["image"]; m=it["message"]; out=[]; g=it["mask128"].ravel()
    for axis in (1,0):
        for a in AS:
            if a==1 and axis==0: continue
            out.append((axis,a,.5)+stat(*cue(X,m,a,axis),g))
    j=int(np.argmin([o[3] for o in out])); ax,a,_,lr,auc=out[j]; res=[]
    if a>1:
        for off in (0,.125,.25,.375,.625,.75,.875,1): res.append((off,)+stat(*cue(X,m,a,ax,off),g))
    print("best %s%.3f lr %.2f auc %.2f (aspect1 lr %.2f auc %.2f) | offsets: "%("W" if ax else "H",a,lr,auc,out[0][3],out[0][4])+" ".join("%.3f:%.2f/%.2f"%r for r in res),flush=True)
    print("     "+" ".join("%s%.2f:%.2f"%("W" if o[0] else "H",o[1],o[3]) for o in out),flush=True)
