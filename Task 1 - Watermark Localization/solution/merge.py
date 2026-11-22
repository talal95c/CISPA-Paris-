# merge children npz (t1_<method>_<hole|patch>.npz in ../kids) with the whole-image type classifier -> local score + submission.pt
# usage: python merge.py [out.pt]
from common import *
import sys
KIDS=ROOT/"kids"; THR={"stegastamp":.75,"trustmark":.78,"pixelseal":.80}
z=np.load("cache/bitacc.npz"); ACC={n:v[0] for n,v in zip(z["names"],z["vals"])}
test_meta=json.load(open(DATA/"test"/"metadata.json"))
def load(meth,ty):
    p=KIDS/f"t1_{meth}_{ty}.npz"
    if not p.exists(): return None
    k=np.load(p,allow_pickle=True)
    return dict(zip([str(n) for n in k["names"]],k["masks"])), dict(zip([str(n) for n in k["train_names"]],k["train_masks"]))
pred={}; tot=[]
for meth in ("stegastamp","trustmark","pixelseal"):
    K={ty:load(meth,ty) for ty in ("hole","patch")}
    tr=ds("train",meth); sc={"hole":[], "patch":[]}; sc_or={"hole":[], "patch":[]}
    for it in tr:
        n=it["image_name"]; ty="hole" if it["full_watermark"] else "patch"
        const=np.full((128,128),int(ty=="hole"),np.uint8)
        oracle=K[ty][1].get(n,const) if K[ty] else const
        tyh="hole" if ACC[n]>=THR[meth] else "patch"
        consth=np.full((128,128),int(tyh=="hole"),np.uint8)
        # a child only has out-of-fold predictions for its own type: a misclassified train image gets the constant mask
        p=oracle if tyh==ty else consth
        sc[ty].append(miou_image(p,it["mask128"])); sc_or[ty].append(miou_image(oracle,it["mask128"]))
    a=sc["hole"]+sc["patch"]; tot+=a
    print(f"{meth}: hole {np.mean(sc['hole']):.4f} (type known {np.mean(sc_or['hole']):.4f})  patch {np.mean(sc['patch']):.4f} (type known {np.mean(sc_or['patch']):.4f})  all {np.mean(a):.4f}   kids: "+", ".join(f"{t}={'yes' if K[t] else 'MISSING'}" for t in K))
    for it in ds("test",meth):
        n=it["image_name"]; tyh="hole" if ACC[n]>=THR[meth] else "patch"
        const=np.full((128,128),int(tyh=="hole"),np.uint8)
        m=K[tyh][0].get(n) if K[tyh] else None
        pred[n]=const if m is None else (np.asarray(m)>0).astype(np.uint8)
print("TRAIN mIoU (out-of-fold, predicted type): %.4f"%np.mean(tot))
if len(sys.argv)>1:
    names=[m["image_name"] for m in test_meta]
    assert len(names)==1020 and set(names)==set(pred)
    M=torch.from_numpy(np.stack([pred[n] for n in names]).astype(bool))
    assert M.shape==(1020,128,128)
    torch.save({"image_names":names,"masks":M},sys.argv[1])
    print("saved",sys.argv[1],os.path.getsize(sys.argv[1]),"bytes; frac ones %.3f"%M.float().mean())
