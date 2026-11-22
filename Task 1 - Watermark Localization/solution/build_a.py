# candidate A: base file, with the mask replaced by the filled bbox of its largest component where that bbox crop decodes the known message (acc >= TH)
from common import *
import pickle, sys
BASE=Path(__file__).resolve().with_name("submission_combo_jureca.pt")
B=torch.load(BASE,weights_only=True); names=list(B["image_names"]); M=B["masks"].clone(); idx={n:i for i,n in enumerate(names)}
bc=pickle.load(open("cache/basecheck_trustmark_pixelseal_stegastamp.pkl","rb")); TH=float(sys.argv[2]) if len(sys.argv)>2 else .9
n=0; ch=[]
for nm,(b,a,fill) in bc.items():
    if a<TH: continue
    m=torch.zeros(128,128,dtype=torch.bool); m[b[0]//4:b[1]//4,b[2]//4:b[3]//4]=True
    d=(m!=M[idx[nm]]).float().mean().item(); ch.append(d); M[idx[nm]]=m; n+=1
print("replaced",n,"mean changed cell frac %.4f"%np.mean(ch),"max %.3f"%np.max(ch), "n changed >1%%: %d"%(np.array(ch)>.01).sum())
torch.save({"image_names":names,"masks":M},sys.argv[1]); print(os.path.getsize(sys.argv[1]))
