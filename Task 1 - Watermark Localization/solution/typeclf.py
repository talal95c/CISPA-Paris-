# type classifier (hole vs patch) from whole-image bit accuracy; constant-mask baseline mIoU on train
from common import *
z=np.load("cache/bitacc.npz"); V=dict(zip(z["names"],z["vals"]))
for meth in ("stegastamp","trustmark","pixelseal"):
    d=ds("train",meth)
    meta=json.load(open(DATA/"train"/"metadata.json"))
    m128=np.load(DATA/"train"/"masks_128.npz")
    rows=[(V[it["image_name"]],it["full_watermark"],it["mask128"]) for it in d]
    acc=np.array([r[0][0] for r in rows]); soft=np.array([r[0][1] for r in rows]); y=np.array([r[1] for r in rows])
    best=None
    for name,f in (("acc",acc),("soft",soft)):
        for th in np.unique(f):
            e=((f>=th)!=y).mean()
            if best is None or e<best[0]: best=(e,name,th)
    print(meth,"best err %.3f on %s thr %.3f"%best)
    for th in (0.65,0.7,0.75,0.8):
        p=acc>=th
        sc=np.mean([miou_image(np.full((128,128),int(pi),np.uint8),r[2]) for pi,r in zip(p,rows)])
        print("   thr %.2f err %.3f  const-mask mIoU %.4f"%(th,(p!=y).mean(),sc))
    te=np.array([V[it["image_name"]][0] for it in ds("test",meth)])
    print("   test acc quantiles",np.round(np.quantile(te,[0,.1,.25,.4,.5,.6,.75,.9,1]),2),"frac>=.75: %.3f"%(te>=.75).mean())
