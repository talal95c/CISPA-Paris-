# frame hypothesis: image = centre crop of a non-square frame that was watermarked whole. pad by reflection to aspect a, embed true/complement msg, crop back, global log energy ratio
from common import *
import cv2, sys
from sklearn.metrics import roc_auc_score
meth=sys.argv[1]; N=int(sys.argv[2]); w=wm(meth); d=ds("train",meth); n=0
AS=[1,5/4,4/3,3/2,16/9,7/4,2]
def cue(X,m,a,axis):
    P=int(round(512*(a-1)/2))
    Xp=X if P==0 else (cv2.copyMakeBorder(X,0,0,P,P,cv2.BORDER_REFLECT_101) if axis==1 else cv2.copyMakeBorder(X,P,P,0,0,cv2.BORDER_REFLECT_101))
    Xf=Xp.astype(np.float32)
    e1=((w.embed(Xp,m).astype(np.float32)-Xf)**2).sum(2); e0=((w.embed(Xp,1-m).astype(np.float32)-Xf)**2).sum(2)
    if P: e1,e0=(e1[:,P:P+512],e0[:,P:P+512]) if axis==1 else (e1[P:P+512],e0[P:P+512])
    return e1,e0
for it in d:
    if not it["full_watermark"] or n>=N: continue
    n+=1; X=it["image"]; m=it["message"]; out=[]; g=it["mask128"].ravel()
    for axis in (1,0):
        for a in AS:
            if a==1 and axis==0: continue
            e1,e0=cue(X,m,a,axis); L=np.log((cv2.blur(e1,(64,64))+.05)/(cv2.blur(e0,(64,64))+.05))
            out.append(("%s%.2f"%("W" if axis==1 else "H",a),np.log(e1.sum()/e0.sum()),roc_auc_score(g,-L.reshape(128,4,128,4).mean((1,3)).ravel())))
    j=int(np.argmin([o[1] for o in out]))
    print("best %s lr %.2f auc64 %.2f | "%out[j]+" ".join("%s:%.2f"%(o[0],o[1]) for o in out),flush=True)
