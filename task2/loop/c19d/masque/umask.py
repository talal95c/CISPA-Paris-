# compare les modeles de masque public: par paire / par endpoint (les deux dans U) / par endpoint (au moins un dans U)
import numpy as np, sys, itertools, time
from mc import eps,ix,pi,pj,NPAIR,Q,obs,FAM,S8M,FP8,K
free=[e for e in eps if e not in S8M and e not in FP8]
nz=1<<len(free)
inO=np.zeros((nz,33),bool)
for e in S8M: inO[:,ix[e]]=True
bits=(np.arange(nz)[:,None]>>np.arange(len(free)))&1
for b,e in enumerate(free): inO[:,ix[e]]=bits[:,b]
T=(inO[:,pi]&inO[:,pj]).astype(np.float32)      # nz x NPAIR
Qf=Q.astype(np.float32)
def feas(m):
    m=m.astype(np.float32);n=m.sum()
    if n<5: return np.zeros(nz,bool)
    Pp=(T@m)[:,None];npred=(m@Qf)[None,:];TP=T@(m[:,None]*Qf)
    FP=npred-TP;FN=Pp-TP;TN=n-npred-FN
    d=npred*Pp*(n-npred)*(n-Pp)
    mcc=np.where(d>0,(TP*TN-FP*FN)/np.sqrt(np.maximum(d,1e-9)),0)
    s=0.1+0.9*np.maximum(mcc,0)
    return (np.abs(s-obs[None,:])<=0.025+1e-6).all(1)
def mk(mode,rng):
    if mode=='pair': return rng.random(NPAIR)<0.3,None
    if mode=='both':
        k=rng.integers(15,23);U=np.zeros(33,bool);U[rng.choice(33,k,replace=False)]=True;return U[pi]&U[pj],U
    if mode=='any':
        k=rng.integers(3,9);U=np.zeros(33,bool);U[rng.choice(33,k,replace=False)]=True;return U[pi]|U[pj],U
if __name__=='__main__':
    mode=sys.argv[1];n=int(sys.argv[2]);seed=int(sys.argv[3]);rng=np.random.default_rng(seed)
    tot=0;hits=[];fr=[];t=time.time()
    for it in range(n):
        m,U=mk(mode,rng);f=feas(m);c=int(f.sum());tot+=c;fr.append(m.mean())
        if c and U is not None: hits.append((c,U.copy(),inO[f].mean(0)))
    print(mode,'masques',n,'| part (Z,masque) compatibles: %.3g'%(tot/(n*nz)),'| masques avec >=1 Z compatible:',len(hits) if mode!='pair' else '-','| part publique moy %.2f'%np.mean(fr),'| %.0fs'%(time.time()-t))
    if hits:
        import pickle;pickle.dump(hits,open(f'umask_{mode}_{seed}.pkl','wb'))
        w=np.array([h[0] for h in hits],float);Us=np.array([h[1] for h in hits]);Pz=np.array([h[2] for h in hits])
        print(' P(endpoint dans U | donnees):',' '.join('%02d:%.2f'%(e,(w@Us[:,ix[e]])/w.sum()) for e in sorted(eps)))
        print(' taille de U:',np.bincount(Us.sum(1),weights=w,minlength=24).round(0)[3:24]/w.sum())
        print(' P(origine | donnees):',' '.join('%02d:%.2f'%(e,(w@Pz[:,ix[e]])/w.sum()) for e in sorted(free)))
