# does the base submission already hold the decodable rectangles? decode the bbox of each base mask (predicted-patch test images)
from common import *
import sys, torch.nn.functional as F
from scipy import ndimage as ndi
from refine import Scorer
B=torch.load(Path(__file__).resolve().with_name("submission_combo_jureca.pt"),weights_only=True)
BM=dict(zip(B["image_names"],B["masks"].numpy().astype(np.uint8)))
z=np.load("cache/bitacc.npz"); ACC={str(n):v[0] for n,v in zip(z["names"],z["vals"])}
THR={"stegastamp":.75,"trustmark":.78,"pixelseal":.80}; out={}
for meth in sys.argv[1:]:
    sc=Scorer(meth); rows=[]
    for it in ds("test",meth):
        n=it["image_name"]
        if ACC[n]>=THR[meth]: continue
        m=BM[n]
        if m.sum()==0: rows.append((n,0,0,0,0)); continue
        lab,k=ndi.label(m); sizes=ndi.sum(m,lab,range(1,k+1)); j=int(np.argmax(sizes))+1
        ys,xs=np.where(lab==j); b=(ys.min()*4,ys.max()*4+4,xs.min()*4,xs.max()*4+4)
        fill=(lab==j).sum()/((b[1]-b[0])*(b[3]-b[2])/16)
        sc.set(it["image"],it["message"]); o,a=sc.score([b])[0]
        rows.append((n,a,fill,(b[1]-b[0])*(b[3]-b[2]),k)); out[n]=(b,a,fill)
    a=np.array([r[1] for r in rows]); f=np.array([r[2] for r in rows])
    print(meth,"pred-patch test images",len(rows),"| base empty",(f==0).sum(),"| base bbox decodes >=.9:",(a>=.9).sum(),">=.75:",(a>=.75).sum(),"| rect-like base (fill>.9):",(f>.9).sum(),"of which decode>=.75:",((f>.9)&(a>=.75)).sum(),flush=True)
import pickle; pickle.dump(out,open("cache/basecheck_%s.pkl"%"_".join(sys.argv[1:]),"wb"))
