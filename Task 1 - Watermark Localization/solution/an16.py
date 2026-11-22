from common import *
import cv2, sys
from sklearn.metrics import roc_auc_score
z=np.load(sys.argv[1],allow_pickle=True); n=len(z["names"]); print("n",n)
def mi(p,g):
    r=[]
    for c in (0,1):
        a,b=(p==c),(g==c); u=(a|b).sum(); r.append(1.0 if u==0 else (a&b).sum()/u)
    return np.mean(r)
def Lmap(e1,e0,k,eps=.05*16): return np.log((cv2.blur(e1,(k,k))+eps)/(cv2.blur(e0,(k,k))+eps))
lr=z["lr"]; lr1=z["lr1"]; fr=z["frame"]
print("lr best quantiles",np.round(np.quantile(lr,[.1,.25,.5,.75,.9]),2),"lr aspect1",np.round(np.quantile(lr1,[.1,.25,.5,.75,.9]),2))
print("aspects",np.round(np.sort(fr[:,1].astype(float)),2)); print("axis",fr[:,0])
for k in (4,8,16,24):
    A=[];
    for i in range(n):
        g=z["gt"][i].astype(bool)
        if g.all() or not g.any(): A.append(np.nan); continue
        A.append(roc_auc_score(g.ravel(),-Lmap(z["e1"][i],z["e0"][i],k).ravel()))
    A=np.array(A); s=lr<-.2
    print("k",k,"AUC all %.3f  strong(lr<-.2, n=%d) %.3f weak %.3f"%(np.nanmean(A),s.sum(),np.nanmean(A[s]),np.nanmean(A[~s])))
# mask rule: clean where L - median(L) > t ; only if lr < g
base=np.mean([mi(np.ones((128,128),bool),z["gt"][i].astype(bool)) for i in range(n)]); print("all-ones mIoU %.4f"%base)
for k in (8,16,24):
    for t in (.3,.5,.7,1.0):
        for gth in (-.1,-.2,-.3):
            sc=[]
            for i in range(n):
                g=z["gt"][i].astype(bool); L=Lmap(z["e1"][i],z["e0"][i],k)
                p=np.ones((128,128),bool)
                if lr[i]<gth: p=~(L-np.median(L)>t)
                sc.append(mi(p,g))
            print("k %d t %.1f gate %.1f mIoU %.4f"%(k,t,gth,np.mean(sc)))
