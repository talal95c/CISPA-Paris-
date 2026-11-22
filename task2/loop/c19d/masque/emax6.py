import numpy as np,itertools
from an import load
from mc import qmat,scores,SUB,K,ix,pi,pj
from s8 import G
A=[(),(17,),(12,),(17,12),(1,),(18,),(24,),(17,1),(12,1),(17,18),(12,18),(1,18),(17,12,1),(17,12,18),(1,18,24)]
D=[(),(7,27,14),(7,27),(27,14),(7,14),(27,),(7,27,14,11),(11,),(7,27,14,5),(7,27,14,28)]
C=[a+d for d in D for a in A]
R={}
for h in ['Uwb10','Uwb12']:
    Z,M=load('samp_%s_*.npz'%h);k=np.random.default_rng(0).choice(len(Z),min(6000,len(Z)),replace=False);Z,M=Z[k],M[k]
    O=Z==0;w=np.where(O[:,ix[11]],0.08/0.2,0.92/0.8);w/=w.sum()
    R[h]=(scores(Z,~M,qmat([G([32]+list(c)) for c in C])),scores(Z,~M,qmat([SUB[k] for k in K])).max(1),w)
for pt in (0.8,0.6):
    W={'Uwb10':pt,'Uwb12':1-pt}
    cur={h:R[h][1].copy() for h in W}
    for c in [(7,27,14),(17,7,27,14),(12,7,27,14)]:
        for h in W: cur[h]=np.maximum(cur[h],R[h][0][:,C.index(c)])
    tot=lambda:sum(W[h]*(R[h][2]*cur[h]).sum() for h in W)
    print('== P(trio vrai)=%.1f ; acquis apres S15, S15+17, S15+12 : %.3f'%(pt,tot()))
    for s in range(2):
        v=sum(W[h]*(R[h][2][:,None]*np.maximum(cur[h][:,None],R[h][0])).sum(0) for h in W);o=np.argsort(-v)
        f=lambda c:('S15+'+' '.join('%02d'%e for e in c if e not in(7,27,14))) if set((7,27,14))<=set(c) else 'S11+'+' '.join('%02d'%e for e in c)
        print('  creneau %d: '%(s+1)+' ; '.join('%s %.4f'%(f(C[j]),v[j]) for j in o[:9]))
        for name in [(17,12,7,27,14),(17,12),(1,7,27,14),(18,7,27,14),(24,7,27,14),(7,27,14,11)]:
            print('      %-14s %.4f'%(f(name),v[C.index(name)]))
        for h in W: cur[h]=np.maximum(cur[h],R[h][0][:,o[0]])
