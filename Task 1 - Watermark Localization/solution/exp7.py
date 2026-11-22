# rect patches: tolerance of crop decoding to scale / shift / aspect errors (bit accuracy vs GT bbox perturbation)
from common import *
import sys
for meth in sys.argv[1:]:
    w=wm(meth); d=ds("train",meth); rows=[]
    P=[("exact",0,0,1,1)]+[("shift x %d%%"%int(100*s),0,s,1,1) for s in (.03,.06,.12,.2)]+[("scale both %.2f"%s,0,0,s,s) for s in (.8,.9,1.1,1.2,1.35)]+[("width x%.2f"%s,0,0,1,s) for s in (.8,1.25)]+[("shift xy 6%",.06,.06,1,1),("shift xy 12%",.12,.12,1,1)]
    for it in d:
        if it["full_watermark"]: continue
        m=it["mask"]>0; ys,xs=np.where(m); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        if m[y0:y1,x0:x1].mean()<=.98: continue
        h,wd=y1-y0,x1-x0; cy,cx=(y0+y1)/2,(x0+x1)/2; x=to_tensor(it["image"]); msg=it["message"]>0; crops=[]
        for _,dy,dx,sh,sw in P:
            a=int(round(cy+dy*h-sh*h/2)); b=int(round(cy+dy*h+sh*h/2)); c=int(round(cx+dx*wd-sw*wd/2)); e=int(round(cx+dx*wd+sw*wd/2))
            a,b,c,e=max(a,0),min(b,512),max(c,0),min(e,512)
            crops.append(float(((w.logits(x[:,:,a:b,c:e])[0].numpy()>0)==msg).mean()) if b-a>3 and e-c>3 else .5)
        rows.append(crops)
    R=np.array(rows); print(meth,len(R))
    for (nm,*_),col in zip(P,R.T): print("   %-16s mean %.3f  frac>=.75: %.2f  frac>=.9: %.2f"%(nm,col.mean(),(col>=.75).mean(),(col>=.9).mean()))
    sys.stdout.flush()
