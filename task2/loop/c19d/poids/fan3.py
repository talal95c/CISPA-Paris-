"""fan3.py g1|g2 file1.npz [file2.npz ...]: cosinus entre empreintes, direction 'temperature' retiree, se par bootstrap sur les reponses."""
import sys,numpy as np,collections
TS=np.array([0.66,0.68,0.70,0.72,0.74,0.76,0.78,0.80,0.84,0.90,1.0])
key=sys.argv[1];E=collections.defaultdict(list)
for f in sys.argv[2:]:
    d=np.load(f);GT=d['GT']
    for k in d.files:
        if k.endswith('|ll'): E[k.split('|')[0]].append((d[k],d[k[:-3]+'|'+key]))
F={};info={};D=0
for e,v in E.items():
    L=np.stack([a for a,_ in v]);n=L[:,-1];T=TS[L[:,:-1].sum(0).argmax()];gi=int(np.abs(GT-T).argmin())
    G=np.stack([g[gi].ravel() for _,g in v]).astype(np.float64);F[e]=(G,n);info[e]=(T,GT[gi],int(n.sum()),len(v))
    D=D+np.stack([(g[-1]-g[0]).ravel() for _,g in v]).sum(0)
eps=sorted(F);sd=np.concatenate([F[e][0]/np.sqrt(F[e][1])[:,None] for e in eps]).std(0)+1e-9
D=D/sd;D/=np.linalg.norm(D)
rng=np.random.default_rng(0)
def fp(e,i):
    G,n=F[e];v=(G[i]/sd).sum(0)/n[i].sum();return v-(v@D)*D
def cosm(idx):
    V=np.stack([fp(e,idx[e]) for e in eps]);V/=np.linalg.norm(V,axis=1)[:,None];return V@V.T
C=cosm({e:np.arange(len(F[e][0])) for e in eps})
B=np.stack([cosm({e:rng.integers(0,len(F[e][0]),len(F[e][0])) for e in eps}) for _ in range(200)]);se=B.std(0)
def selfz(e):
    G,n=F[e];G=G/sd;G=G-np.outer(G@D,D);S=G.sum(0);tot=S@S-(G*G).sum()
    x=np.array([(S-g)@(S-g)-((G*G).sum()-g@g) for g in G]);return tot/np.sqrt((len(G)-1)/len(G)*((x-x.mean())**2).sum())
for e in eps: print(e,'T*=%.2f GT=%.2f tok=%d n=%d'%info[e],'puissance propre z=%.1f'%selfz(e))
print('cosinus (se bootstrap) ; biais vers le bas du bootstrap non corrige')
print('      '+' '.join('%12s'%e[-2:] for e in eps))
for p,x in enumerate(eps): print(x[-5:],' '.join('     --     ' if p==q else '%+.3f(%.3f)'%(C[p,q],se[p,q]) for q in range(len(eps))))
