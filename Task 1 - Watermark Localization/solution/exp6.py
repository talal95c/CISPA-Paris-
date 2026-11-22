# pixelseal: detection channel (0) and message agreement per position, at input res 256 / 512 / 768 (fully conv)
from common import *
import cv2, sys, time
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
w=wm("pixelseal"); d=ds("train","pixelseal"); N=int(sys.argv[1]) if len(sys.argv)>1 else 8
rng=np.random.default_rng(2); R={True:[],False:[]}; t=time.time(); n=0
def maps(x,res):
    xr=F.interpolate(x,size=(res,res),mode="bilinear",align_corners=False,antialias=True)
    with torch.no_grad():
        lat=w.det.convnext(w.det.preprocess(xr)); emb=w.det.pixel_decoder.output_upscaling(lat); lin=w.det.pixel_decoder.linear
        return torch.einsum("bchw,oc->bohw",emb,lin.weight)+lin.bias[None,:,None,None]
for i in rng.permutation(len(d)):
    it=d[int(i)]
    if len(R[it["full_watermark"]])>=N: continue
    n+=1; x=to_tensor(it["image"]); s=torch.tensor(it["message"]*2.0-1)[None,:,None,None]; g=it["mask128"].ravel(); row=[]
    for res in (256,512,768):
        pp=maps(x,res)
        for mp in (pp[:,0], (pp[:,1:]*s).mean(1), ((pp[:,1:]*s)>0).float().mean(1)):
            up=F.interpolate(mp[None],size=(128,128),mode="bilinear",align_corners=False)[0,0].numpy()
            row.append(roc_auc_score(g,up.ravel()))
    R[it["full_watermark"]].append(row)
    if n==1: print("shape",pp.shape,"%.1fs"%(time.time()-t),flush=True)
print("%.2fs/img"%((time.time()-t)/n))
for k,v in R.items():
    v=np.array(v).reshape(len(v),3,3); print(" full=",k,"rows res 256/512/768, cols det-chan / soft agree / hard agree"); print(np.round(v.mean(0),3)); print(np.round(np.median(v,0),3))
