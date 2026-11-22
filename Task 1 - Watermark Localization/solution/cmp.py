# compare a kid npz with the base file on the test images of its slice (no labels): pairwise mIoU agreement, and kid's train score
from common import *
import sys
meth,typ,f=sys.argv[1:4]
B=torch.load(Path(__file__).resolve().with_name("submission_combo_jureca.pt"),weights_only=True)
BM=dict(zip(B["image_names"],B["masks"].numpy().astype(np.uint8)))
z=np.load("cache/bitacc.npz"); ACC={str(n):v[0] for n,v in zip(z["names"],z["vals"])}
THR={"stegastamp":.75,"trustmark":.78,"pixelseal":.80}
k=np.load(f,allow_pickle=True); print({x:k[x].shape for x in k.files})
KM=dict(zip([str(n) for n in k["names"]],(k["masks"]>0).astype(np.uint8)))
ag=[];fb=[];fk=[]
for n,m in KM.items():
    hole=ACC[n]>=THR[meth]
    if hole!=(typ=="hole"): continue
    ag.append(miou_image(m,BM[n])); fb.append(BM[n].mean()); fk.append(m.mean())
ag=np.array(ag); print("test slice n",len(ag),"agreement mIoU kid vs base: mean %.3f median %.3f"%(ag.mean(),np.median(ag)),"| mean pos frac base %.3f kid %.3f"%(np.mean(fb),np.mean(fk)))
if "train_masks" in k.files:
    d={it["image_name"]:it for it in ds("train",meth)}
    s=[miou_image((m>0).astype(np.uint8),d[str(n)]["mask128"]) for n,m in zip(k["train_names"],k["train_masks"]) if d[str(n)]["full_watermark"]==(typ=="hole")]
    print("kid train slice n",len(s),"mIoU %.4f"%np.mean(s))
