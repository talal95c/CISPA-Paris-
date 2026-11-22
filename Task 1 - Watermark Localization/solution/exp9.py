# trustmark: remover-based matched filter. a = X - remove(X) (present watermark estimate), b = embed(remove(X),msg) - remove(X) (expected watermark). coef = <a,b>/<b,b> locally.
from common import *
import cv2, sys, time
from PIL import Image
from sklearn.metrics import roc_auc_score
w=wm("trustmark"); tm=w.tm; N=int(sys.argv[1]) if len(sys.argv)>1 else 10
print("has removal:",hasattr(tm,"removal"),flush=True)
d=ds("train","trustmark"); rng=np.random.default_rng(3); R={True:[],False:[]}; t=time.time(); n=0
for i in rng.permutation(len(d)):
    it=d[int(i)]
    if len(R[it["full_watermark"]])>=N: continue
    n+=1; X=it["image"]; Xf=X.astype(np.float32); msg=it["message"]
    Xr=np.asarray(tm.remove_watermark(Image.fromarray(X)).convert("RGB"))
    a=Xf-Xr.astype(np.float32)
    b=w.embed(Xr,msg).astype(np.float32)-Xr.astype(np.float32)
    b2=w.embed(X,msg).astype(np.float32)-Xf
    m=it["mask"]>0; g=it["mask128"].ravel(); row=[]
    for (p,q) in ((a,b),(a,b2)):
        num=(p*q).sum(2); den=(q*q).sum(2)
        for win in (8,16,32,64):
            mp=(cv2.blur(num,(win,win))/(cv2.blur(den,(win,win))+.05)).reshape(128,4,128,4).mean((1,3))
            row.append(roc_auc_score(g,mp.ravel()))
        row+=[num[m].sum()/(den[m].sum()+1e-6), num[~m].sum()/(den[~m].sum()+1e-6)]
    R[it["full_watermark"]].append(row)
print("%.2fs/img"%((time.time()-t)/n))
for k,v in R.items():
    v=np.array(v); print(" full=",k)
    for j,nm in enumerate(("b=embed(removed)","b=embed(X)")):
        u=v[:,6*j:6*j+6]
        print("   %-17s AUC win 8/16/32/64 mean %s median %s | coef marked %.3f sd %.3f, clean %.3f sd %.3f, marked>clean %.2f"%(nm,np.round(u[:,:4].mean(0),3),np.round(np.median(u[:,:4],0),3),u[:,4].mean(),u[:,4].std(),u[:,5].mean(),u[:,5].std(),(u[:,4]>u[:,5]).mean()))
