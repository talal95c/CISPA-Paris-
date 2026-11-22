# (1) is the embed residual stable under re-embedding?  (2) region-level matched filter: ratio = <hp(X), hp(r_hat)> / <hp(r_hat),hp(r_hat)> on GT marked vs clean region
from common import *
import cv2, sys
meth=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 12
w=wm(meth); d=ds("train",meth)
def hp(x,s): return x-cv2.GaussianBlur(x,(0,0),s)
def cc(a,b): a=a-a.mean(); b=b-b.mean(); return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()+1e-9))
rng=np.random.default_rng(0)
C=[]; R={True:[],False:[]}
for i in rng.permutation(len(d))[:2*N]:
    it=d[int(i)]; X=it["image"]; Xf=X.astype(np.float32); msg=it["message"]
    W1=w.embed(X,msg); r1=W1.astype(np.float32)-Xf
    if len(C)<6:
        W2=w.embed(W1,msg); r2=W2.astype(np.float32)-W1
        other=rng.integers(0,2,len(msg)); r3=w.embed(X,other).astype(np.float32)-Xf
        C.append((cc(r1,r2),cc(r1,r3),np.abs(r1).mean(),np.abs(r2).mean()))
    m=it["mask"]>0
    row=[]
    for s in (0.7,1.5,3.0):
        a=hp(Xf,s); b=hp(r1,s); num=(a*b).sum(2); den=(b*b).sum(2)
        for reg in (m,~m):
            row.append(num[reg].sum()/(den[reg].sum()+1e-6) if reg.sum()>200 else np.nan)
    R[it["full_watermark"]].append(row)
C=np.array(C); print(meth,"corr(r(X),r(W(X))) %.3f  corr same img other msg %.3f  |r1| %.2f |r2| %.2f"%tuple(C.mean(0)))
for k,v in R.items():
    v=np.array(v); print(" full=",k,"n",len(v))
    for j,s in enumerate((0.7,1.5,3.0)):
        a,b=v[:,2*j],v[:,2*j+1]
        print("   hp s=%.1f ratio marked: mean %.3f sd %.3f | clean: mean %.3f sd %.3f | frac images marked>clean %.2f"%(s,np.nanmean(a),np.nanstd(a),np.nanmean(b),np.nanstd(b),np.nanmean((a>b)[~np.isnan(a)&~np.isnan(b)])))
