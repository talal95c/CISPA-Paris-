# keep-only occlusion: paste one tile of X on a neutral background at its original position/scale, decode, agreement with the known message
from common import *
import sys, time, cv2
from sklearn.metrics import roc_auc_score
meth=sys.argv[1]; N=int(sys.argv[2]); w=wm(meth); d=ds("train",meth); rng=np.random.default_rng(5)
R={True:[],False:[]}; t=time.time(); n=0
for i in rng.permutation(len(d)):
    it=d[int(i)]
    if len(R[it["full_watermark"]])>=N: continue
    n+=1; x=to_tensor(it["image"]); s=torch.tensor(it["message"]*2.0-1); g=it["mask128"].ravel(); row=[]
    bgs={"gray":torch.full_like(x,.5),"blur":torch.tensor(cv2.GaussianBlur(it["image"],(0,0),12)).permute(2,0,1)[None].float()/255}
    for bn,bg in bgs.items():
        for T in (64,128,192):
            st=T//2; acc=np.zeros((512,512),np.float32); cnt=np.zeros((512,512),np.float32); batch=[]; pos=[]
            P=sorted(set(list(range(0,512-T+1,st))+[512-T]))
            for y in P:
                for xx in P:
                    c=bg.clone(); c[:,:,y:y+T,xx:xx+T]=x[:,:,y:y+T,xx:xx+T]; batch.append(c); pos.append((y,xx))
            zs=[]
            for j in range(0,len(batch),32):
                with torch.no_grad(): zs.append((w.logits(torch.cat(batch[j:j+32]))*s).mean(1))
            zs=torch.cat(zs).numpy()
            for (y,xx),v in zip(pos,zs): acc[y:y+T,xx:xx+T]+=v; cnt[y:y+T,xx:xx+T]+=1
            mp=(acc/cnt).reshape(128,4,128,4).mean((1,3)); row.append(roc_auc_score(g,mp.ravel()))
    R[it["full_watermark"]].append(row)
print(meth,"%.1fs/img"%((time.time()-t)/n),"cols: gray 64/128/192, blur 64/128/192")
for k,v in R.items(): v=np.array(v); print(" full=",k,"AUC mean",np.round(v.mean(0),3),"median",np.round(np.median(v,0),3))
