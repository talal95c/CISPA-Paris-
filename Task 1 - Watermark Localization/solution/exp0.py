# whole-image bit accuracy per type
from common import *
import time
for meth in ("stegastamp","trustmark","pixelseal"):
    w=wm(meth); d=ds("train",meth); t=time.time()
    acc={True:[],False:[]}
    for it in d:
        b=w.decode(to_tensor(it["image"]))[0].numpy()
        acc[it["full_watermark"]].append((b==it["message"]).mean())
    print(meth, "time/img %.3f"%((time.time()-t)/len(d)))
    for k,v in acc.items(): print("  full=",k,np.round(np.quantile(v,[0,.1,.25,.5,.75,.9,1]),2))
