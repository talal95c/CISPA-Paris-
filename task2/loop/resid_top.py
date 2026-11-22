import numpy as np, json
from resid_an import loadres, agg
from transformers import AutoTokenizer
tok=AutoTokenizer.from_pretrained('/home/ubuntu/models/HuggingFaceTB__SmolLM2-360M-Instruct')
keys,O,E,V=loadres('wb/res_r360.npz')
for name,sel in (('short',lambda k: not k[1].startswith('A_L')),('long',lambda k:k[1].startswith('A_L'))):
    A=agg(keys,O,E,V,sel)
    for e in ('ref-01','ref-02','ref-09','ref-04','ref-07','ref-08'):
        R,Vv,n=A[e]; z=R/np.sqrt(Vv+0.5); i=np.argsort(-np.abs(z))[:10]
        print(name,e,n,' '.join(f'{tok.decode([j])!r}:{R[j]:+.0f}/{z[j]:+.1f}' for j in i))
