# message template matched filter (idea from the trustmark-patch child): T = mean_k [embed(cover_k,msg)-cover_k] over other covers; local correlation of high-passed X with T.
from common import *
import cv2, sys, time
from sklearn.metrics import roc_auc_score
meth=sys.argv[1]; N=int(sys.argv[2]); K=int(sys.argv[3]) if len(sys.argv)>3 else 8
w=wm(meth); d=ds("train",meth); rng=np.random.default_rng(4); R={True:[],False:[]}; t=time.time(); n=0
def hp(x,s): return x-cv2.GaussianBlur(x,(0,0),s)
covers=[d[int(i)]["image"] for i in rng.permutation(len(d))[:K]]
for i in rng.permutation(len(d)):
    it=d[int(i)]
    if len(R[it["full_watermark"]])>=N: continue
    n+=1; X=it["image"].astype(np.float32); msg=it["message"]
    T=np.mean([w.embed(c,msg).astype(np.float32)-c for c in covers],0)
    m=it["mask"]>0; g=it["mask128"].ravel(); row=[]
    for s in (1.5,4.0):
        a=hp(X,s); b=hp(T,s); num=(a*b).sum(2); den=(b*b).sum(2); ea=(a*a).sum(2)
        for win in (8,16,32,64):
            nn=cv2.blur(num,(win,win)); mp=(nn/np.sqrt(cv2.blur(den,(win,win))*cv2.blur(ea,(win,win))+1e-6)).reshape(128,4,128,4).mean((1,3))
            row.append(roc_auc_score(g,mp.ravel()))
        row+=[num[m].sum()/(den[m].sum()+1e-6), num[~m].sum()/(den[~m].sum()+1e-6)]
    R[it["full_watermark"]].append(row)
print(meth,"K",K,"%.2fs/img"%((time.time()-t)/n))
for k,v in R.items():
    v=np.array(v); print(" full=",k)
    for j,s in enumerate((1.5,4.0)):
        u=v[:,6*j:6*j+6]
        print("   hp %.1f AUC win 8/16/32/64 mean %s median %s | coef marked %.3f sd %.3f, clean %.3f sd %.3f, marked>clean %.2f"%(s,np.round(u[:,:4].mean(0),3),np.round(np.median(u[:,:4],0),3),u[:,4].mean(),u[:,4].std(),u[:,5].mean(),u[:,5].std(),(u[:,4]>u[:,5]).mean()))
