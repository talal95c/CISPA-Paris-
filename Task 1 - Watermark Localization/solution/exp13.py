# trustmark holes: log energy ratio of re-embedding with true vs complement message (idea from trust_hole kid), visual + AUC
from common import *
import cv2, sys
from sklearn.metrics import roc_auc_score
meth=sys.argv[1]; N=int(sys.argv[2]); w=wm(meth); d=ds("train",meth); tiles=[]; A=[]; n=0
for it in d:
    if not it["full_watermark"] or n>=N: continue
    n+=1; X=it["image"]; Xf=X.astype(np.float32); m=it["message"]
    e1=((w.embed(X,m).astype(np.float32)-Xf)**2).sum(2); e0=((w.embed(X,1-m).astype(np.float32)-Xf)**2).sum(2)
    row=[]
    for win in (16,32,64):
        L=np.log((cv2.blur(e1,(win,win))+.05)/(cv2.blur(e0,(win,win))+.05)); row.append(roc_auc_score(it["mask128"].ravel(),-L.reshape(128,4,128,4).mean((1,3)).ravel()))
    A.append(row+[np.log(e1.sum()/e0.sum()), it["mask128"].mean()])
    L=np.log((cv2.blur(e1,(32,32))+.05)/(cv2.blur(e0,(32,32))+.05))
    v=np.clip((L+3)/6*255,0,255).astype(np.uint8); hm=np.ascontiguousarray(cv2.applyColorMap(v,cv2.COLORMAP_JET)[...,::-1])
    Xc=X.copy(); c=cv2.findContours((it["mask"]==0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    cv2.drawContours(Xc,c,-1,(255,0,0),3); cv2.drawContours(hm,c,-1,(255,255,255),3)
    tiles.append(np.hstack([cv2.resize(Xc,(192,192)),cv2.resize(hm,(192,192))]))
A=np.array(A); print("AUC 16/32/64 mean",np.round(A[:,:3].mean(0),3),"median",np.round(np.median(A[:,:3],0),3))
for a in A: print("  auc %.2f %.2f %.2f  global logratio %.2f  marked frac %.2f"%tuple(a))
S=np.vstack([np.hstack(tiles[i*4:(i+1)*4]) for i in range(len(tiles)//4)]); cv2.imwrite("sheet13_%s.jpg"%meth,S[...,::-1],[cv2.IMWRITE_JPEG_QUALITY,80])
