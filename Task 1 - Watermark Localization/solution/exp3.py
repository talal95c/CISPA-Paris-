# patch type: does decoding the GT bounding-box crop give back the message? (vs whole image, vs bbox padded/shifted)
from common import *
import sys
for meth in sys.argv[1:]:
    w=wm(meth); d=ds("train",meth); R=[]
    for it in d:
        if it["full_watermark"]: continue
        m=it["mask"]>0; ys,xs=np.where(m)
        if len(ys)==0: continue
        y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        fill=m[y0:y1,x0:x1].mean()
        x=to_tensor(it["image"]); msg=it["message"]>0
        def acc(a,b,c,e):
            a,b,c,e=max(a,0),min(b,512),max(c,0),min(e,512)
            return float(((w.logits(x[:,:,a:b,c:e])[0].numpy()>0)==msg).mean())
        h,wd=y1-y0,x1-x0
        R.append((fill,h,wd,acc(y0,y1,x0,x1),acc(y0-h//4,y1+h//4,x0-wd//4,x1+wd//4),acc(y0+h//4,y1-h//4,x0+wd//4,x1-wd//4),acc(y0,y1,x0+wd//4,x1+wd//4),acc(0,512,0,512)))
    R=np.array(R); print(meth,"n",len(R))
    for nm,sel in (("rect(fill>.98)",R[:,0]>.98),("object",R[:,0]<=.98)):
        r=R[sel]; print(" ",nm,len(r),"bbox %.3f pad25 %.3f shrink25 %.3f shift25 %.3f whole %.3f"%tuple(r[:,3:].mean(0)), "| bbox acc q10/50/90",np.round(np.quantile(r[:,3],[.1,.5,.9]),2))
    small=R[:,1]*R[:,2]<64*64
    print("  bbox area<64^2:",small.sum(),"bbox acc %.3f"%R[small,3].mean(), " larger: %.3f"%R[~small,3].mean(),flush=True)
