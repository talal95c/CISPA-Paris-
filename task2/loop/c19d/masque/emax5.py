import numpy as np,itertools,sys,os
from an import load
from mc import qmat,scores,SUB,K,ix,pi,pj
from s8 import G
A=[(),(17,),(12,),(17,12),(1,),(18,),(24,)]
D=[7,27,14,11,5,28,2]
C=[tuple(a)+tuple(d) for r in range(8) for d in itertools.combinations(D,r) for a in A]
POOL=[17,25,12,18,24,1,14,7,27,11,5,2,28,3,21]
R={}
for h in ['Uwb9','Uwb10','Uwb11','Uwb12']:
    Z,M=load('samp_%s_*.npz'%h);k=np.random.default_rng(0).choice(len(Z),min(4000,len(Z)),replace=False);Z,M=Z[k],M[k]
    Q=qmat([G([32]+list(c)) for c in C]);hid=scores(Z,~M,Q)
    base=scores(Z,~M,qmat([SUB[k] for k in K])).max(1);O=Z==0
    print('==',h,len(Z),'max cache acquis %.3f'%base.mean(),'| P(orig):',' '.join('%02d:%.2f'%(e,O[:,ix[e]].mean()) for e in POOL),'| |U| %.1f'%0)
    R[h]=(hid,base)
PEND=(7,27,14)  # cree a 08:40 UTC
def run(W,name,n=4):
    print('==',name,W)
    cur={h:np.maximum(R[h][1],R[h][0][:,C.index(PEND)]) for h in W}
    tot=lambda:sum(W[h]*cur[h].mean() for h in W)
    print('  apres S11+07 27 14 (08:40): %.3f'%tot())
    for s in range(n):
        v=sum(W[h]*np.maximum(cur[h][:,None],R[h][0]).mean(0) for h in W);o=np.argsort(-v)
        print('  %d. S11+%-22s E[max] %.3f (+%.3f) | suivants: %s'%(s+1,' '.join('%02d'%e for e in C[o[0]]),v[o[0]],v[o[0]]-tot(),' ; '.join('%s %.3f'%(' '.join('%02d'%e for e in C[j]),v[j]) for j in o[1:4])))
        for h in W: cur[h]=np.maximum(cur[h],R[h][0][:,o[0]])
run({'Uwb9':0.8,'Uwb11':0.2},'cas 05/28 temperature pure (0,7)')
run({'Uwb10':0.8,'Uwb12':0.2},'cas 05/28 LoRA (0,1)')
run({'Uwb9':0.4,'Uwb11':0.1,'Uwb10':0.4,'Uwb12':0.1},'melange des deux cas')
