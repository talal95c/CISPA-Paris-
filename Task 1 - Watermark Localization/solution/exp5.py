# re-embedding energy ratio: |r(X,msg)|^2 vs |r(X,other msgs)|^2, locally. marked regions should need a smaller residual for the message they already carry.
from common import *
import cv2, sys, time
from sklearn.metrics import roc_auc_score
meth=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 10; K=3
w=wm(meth); d=ds("train",meth)
rng=np.random.default_rng(1)
R={True:[],False:[]}; t=time.time(); n=0
for i in rng.permutation(len(d)):
    it=d[int(i)]
    if len(R[it["full_watermark"]])>=N: continue
    X=it["image"]; Xf=X.astype(np.float32); msg=it["message"]; n+=1
    e=((w.embed(X,msg).astype(np.float32)-Xf)**2).sum(2)
    eo=np.mean([((w.embed(X,rng.integers(0,2,len(msg))).astype(np.float32)-Xf)**2).sum(2) for _ in range(K)],0)
    m=it["mask"]>0; g=it["mask128"].ravel(); row=[]
    for win in (8,16,32,64):
        a=cv2.blur(e,(win,win)); b=cv2.blur(eo,(win,win))
        mp=-np.log((a+.05)/(b+.05)).reshape(128,4,128,4).mean((1,3))
        row.append(roc_auc_score(g,mp.ravel()))
    row+= [e[m].mean()/(eo[m].mean()+1e-6), e[~m].mean()/(eo[~m].mean()+1e-6)]
    R[it["full_watermark"]].append(row)
print(meth,"%.2fs/img"%((time.time()-t)/n))
for k,v in R.items():
    v=np.array(v); print(" full=",k,"AUC win 8/16/32/64 mean",np.round(v[:,:4].mean(0),3),"median",np.round(np.median(v[:,:4],0),3))
    print("    region energy ratio marked: %.3f sd %.3f | clean: %.3f sd %.3f | frac marked<clean %.2f"%(v[:,4].mean(),v[:,4].std(),v[:,5].mean(),v[:,5].std(),(v[:,4]<v[:,5]).mean()))
