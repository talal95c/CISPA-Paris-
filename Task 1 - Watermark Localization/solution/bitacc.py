# whole-image bit accuracy + soft margin for every train/test image -> cache/bitacc.npz (type classifier input)
from common import *
import time
out={}
for meth in ("stegastamp","trustmark","pixelseal"):
    w=wm(meth)
    for split in ("train","test"):
        d=ds(split,meth); t=time.time()
        for i in range(0,len(d),8):
            its=[d[j] for j in range(i,min(i+8,len(d)))]
            x=torch.cat([to_tensor(it["image"]) for it in its])
            lg=w.logits(x).numpy()
            for it,l in zip(its,lg):
                s=it["message"]*2.0-1
                out[it["image_name"]]=(((l>0)==(it["message"]>0)).mean(), float((l*s).mean()), float(np.abs(l).mean()))
        print(meth,split,len(d),"%.1fs"%(time.time()-t),flush=True)
os.makedirs("cache",exist_ok=True)
np.savez("cache/bitacc.npz",names=np.array(list(out)),vals=np.array(list(out.values())))
