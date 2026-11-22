import numpy as np, json
from resid_an import loadres, agg
from wb_fit import kind
keys,O,E,V=loadres('wb/res_r360.npz')
ids=sorted({k[1] for k in keys}); h={i:n%2 for n,i in enumerate(ids)}
sels={'short':lambda k: not k[1].startswith('A_L'),'long':lambda k:k[1].startswith('A_L'),'half0':lambda k:h[k[1]]==0,'half1':lambda k:h[k[1]]==1}
for e in sorted(kind,key=lambda e:kind[e]):
    if not e.startswith('ref'): continue
    s=f'{e} {kind[e][:16]:<17}'
    for name,sel in sels.items():
        R,Vv,n=agg(keys,O,E,V,lambda k: k[0]==e and sel(k))[e]
        m=sum(1 for k in keys if k[0]==e and sel(k))
        s+=f'| {name} n={m:3d} tok={n:5d} Eeos={-R[2]:6.1f} z={R[2]/np.sqrt(Vv[2]):6.1f} dot={R[30]:+5.0f}/{R[30]/np.sqrt(Vv[30]+.5):+5.1f} '
    print(s)
