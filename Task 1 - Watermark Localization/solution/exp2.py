# per-position logit agreement maps, whole image + tiled crops
from common import *
import cv2, sys, time
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
meth=sys.argv[1]; N=int(sys.argv[2])
w=wm(meth); d=ds("train",meth)
def agree_maps(x, msg):
    with torch.no_grad(): pp=w.features(x)["per_position_logits"]  # B,nbits,h,w
    s=torch.tensor(msg*2.0-1)[None,:,None,None]
    soft=(pp*s).mean(1); hard=((pp*s)>0).float().mean(1)
    return soft, hard
def tiled(I, msg, T, S):
    x=to_tensor(I); crops=[]; pos=[]
    for y in range(0,512-T+1,S):
        for xx in range(0,512-T+1,S):
            crops.append(x[:,:,y:y+T,xx:xx+T]); pos.append((y,xx))
    soft,hard=agree_maps(torch.cat(crops),msg)
    out=[]
    for m in (soft,hard):
        acc=torch.zeros(128,128); cnt=torch.zeros(128,128)
        mm=F.interpolate(m[:,None],size=(T//4,T//4),mode="bilinear",align_corners=False)[:,0]
        for k,(y,xx) in enumerate(pos):
            acc[y//4:(y+T)//4, xx//4:(xx+T)//4]+=mm[k]; cnt[y//4:(y+T)//4, xx//4:(xx+T)//4]+=1
        out.append((acc/cnt).numpy())
    return out
res={}; t=time.time(); n=0
cfgs=[(512,512),(256,128),(128,64)]
for i in range(len(d)):
    it=d[i]
    if len(res.setdefault(it["full_watermark"],[]))>=N: continue
    g=it["mask128"].ravel(); row=[]
    for T,S in cfgs:
        for m in tiled(it["image"],it["message"],T,S):
            row.append(roc_auc_score(g,m.ravel()) if 0<g.sum()<g.size else np.nan)
    res[it["full_watermark"]].append(row); n+=1
print("shape per-pos", w.features(to_tensor(d[0]["image"]))["per_position_logits"].shape, "time/img", (time.time()-t)/n)
for k,v in res.items():
    v=np.array(v); print(meth,"full=",k,"cols: (soft,hard) x",cfgs); print(" mean",np.round(np.nanmean(v,0),3)); print(" med ",np.round(np.nanmedian(v,0),3))
