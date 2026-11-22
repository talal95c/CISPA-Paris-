# exp16.py meth split typ N out : per image frame search (centre crop of taller/wider frame), save e1/e0 128x128 cell sums in best frame
from common import *
import cv2, sys, time
meth,split,typ,N,out=sys.argv[1],sys.argv[2],sys.argv[3],int(sys.argv[4]),sys.argv[5]
w=wm(meth); d=ds(split,meth)
z=np.load("cache/bitacc.npz"); ACC={str(n):v[0] for n,v in zip(z["names"],z["vals"])}
THR={"stegastamp":.75,"trustmark":.78,"pixelseal":.80}
def cue(X,m,a,axis,off=.5):
    T=int(round(512*(a-1))); P0=int(round(T*off)); P1=T-P0
    Xp=X if T==0 else (cv2.copyMakeBorder(X,0,0,P0,P1,cv2.BORDER_REFLECT_101) if axis==1 else cv2.copyMakeBorder(X,P0,P1,0,0,cv2.BORDER_REFLECT_101))
    Xf=Xp.astype(np.float32)
    e1=((w.embed(Xp,m).astype(np.float32)-Xf)**2).sum(2); e0=((w.embed(Xp,1-m).astype(np.float32)-Xf)**2).sum(2)
    if T: e1,e0=(e1[:,P0:P0+512],e0[:,P0:P0+512]) if axis==1 else (e1[P0:P0+512],e0[P0:P0+512])
    return e1,e0
def glob(e1,e0,typ):
    if typ=="hole": return float(np.log(e1.sum()/e0.sum()))
    L=np.log((cv2.blur(e1,(96,96))+.05)/(cv2.blur(e0,(96,96))+.05)); return float(L.min())
HYP=[(1,1.0,.5)]+[(0,a,.5) for a in (1.07,1.14,1.2,1.27,1.33,1.4,1.5,1.6,1.75)]+[(1,a,.5) for a in (1.33,1.78)]
SH=[int(v) for v in os.environ.get("SH","0/1").split("/")]; cnt=0
R={k:[] for k in "names e1 e0 frame lr lr1 gt e1a e0a".split()}; t=time.time(); n=0
for it in d:
    nm=it["image_name"]
    ish = it["full_watermark"] if split=="train" else ACC[nm]>=THR[meth]
    if ish!=(typ=="hole") or n>=N: continue
    cnt+=1
    if cnt%SH[1]!=SH[0]: continue
    n+=1; X=it["image"]; m=it["message"]; res={}
    for h in HYP: res[h]=cue(X,m,h[1],h[0],h[2])
    sc={h:glob(*res[h],typ) for h in HYP}; b=min(sc,key=sc.get)
    if b[1]>1:
        for h in [(b[0],b[1]-.035,.5),(b[0],b[1]+.035,.5),(b[0],b[1],.42),(b[0],b[1],.58)]:
            res[h]=cue(X,m,h[1],h[0],h[2]); sc[h]=glob(*res[h],typ)
        b=min(sc,key=sc.get)
    cs=lambda e: e.reshape(128,4,128,4).sum((1,3)).astype(np.float32)
    R["names"].append(nm); R["e1"].append(cs(res[b][0])); R["e0"].append(cs(res[b][1])); R["frame"].append(b); R["lr"].append(sc[b]); R["lr1"].append(sc[HYP[0]])
    R["e1a"].append(cs(res[HYP[0]][0])); R["e0a"].append(cs(res[HYP[0]][1]))
    R["gt"].append(it["mask128"] if split=="train" else np.zeros((128,128),bool))
    print(n,nm,b,"%.2f %.2f"%(sc[b],sc[HYP[0]]),"%.1fs/img"%((time.time()-t)/n),flush=True)
    if n%10==0: np.savez_compressed(out,**{k:np.array(v) for k,v in R.items()})
np.savez_compressed(out,**{k:np.array(v) for k,v in R.items()})
