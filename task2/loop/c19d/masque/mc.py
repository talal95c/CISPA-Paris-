import numpy as np, sys, itertools, time
from scipy.special import gammaln
from parts2 import SUB,OBS,K,FAM
eps=[e for f in 'ABC' for e in FAM[f]]
ix={e:i for i,e in enumerate(eps)}
nf=[len(FAM[f]) for f in 'ABC']; off=[0,nf[0],nf[0]+nf[1]]
famof=np.repeat(np.arange(3),nf); local=np.concatenate([np.arange(n) for n in nf])
pairs=[(ix[a],ix[b]) for f in 'ABC' for a,b in itertools.combinations(FAM[f],2)]
pi=np.array([p[0] for p in pairs]);pj=np.array([p[1] for p in pairs]);NPAIR=len(pairs)
def qmat(subs):
    Q=np.zeros((NPAIR,len(subs)),np.float32)
    for k,gs in enumerate(subs):
        lab=np.arange(33)+100
        for gi,g in enumerate(gs):
            for e in g: lab[ix[e]]=gi
        Q[:,k]=(lab[pi]==lab[pj])
    return Q
Q=qmat([SUB[k] for k in K]);obs=np.array([OBS[k] for k in K],np.float32)
def scores(Z,M,Q=Q):
    T=Z[:,pi]==Z[:,pj];MT=M&T
    n=M.sum(1,dtype=np.float32)[:,None];Pp=MT.sum(1,dtype=np.float32)[:,None]
    npred=M.astype(np.float32)@Q;TP=MT.astype(np.float32)@Q
    FP=npred-TP;FN=Pp-TP;TN=n-npred-FN
    d=npred*Pp*(n-npred)*(n-Pp)
    mcc=np.where(d>0,(TP*TN-FP*FN)/np.sqrt(np.maximum(d,1e-9)),0)
    return 0.1+0.9*np.maximum(mcc,0)
import os as _os
_RND=_os.environ.get('RND','near');_AGG=_os.environ.get('AGG','glob')
_pf=None
def scores_fam(Z,M):
    global _pf
    if _pf is None: _pf=[famof[pi]==f for f in range(3)]
    T=Z[:,pi]==Z[:,pj];tot=0
    for f in range(3):
        Mf=M[:,_pf[f]];Tf=T[:,_pf[f]];Qf=Q[_pf[f]];MT=Mf&Tf
        n=Mf.sum(1,dtype=np.float32)[:,None];Pp=MT.sum(1,dtype=np.float32)[:,None]
        npred=Mf.astype(np.float32)@Qf;TP=MT.astype(np.float32)@Qf
        FP=npred-TP;FN=Pp-TP;TN=n-npred-FN;d=npred*Pp*(n-npred)*(n-Pp)
        mcc=np.where(d>0,(TP*TN-FP*FN)/np.sqrt(np.maximum(d,1e-9)),np.where((FP+FN)==0,1.0,0.0))
        tot=tot+np.maximum(mcc,0)
    return 0.1+0.9*tot/3
def energy(Z,M,tol=float(_os.environ.get("TOL",0.025))):
    if _AGG in('diag','cross','diagcross'):
        T=Z[:,pi]==Z[:,pj];MT=M&T;u=_CU.sum(1).astype(np.float32)[:,None]
        n=M.sum(1,dtype=np.float32)[:,None];Pp=MT.sum(1,dtype=np.float32)[:,None]
        npred=M.astype(np.float32)@Q;TP=MT.astype(np.float32)@Q
        FP=npred-TP;FN=Pp-TP;TN=n-npred-FN
        if 'cross' in _AGG: TN=TN+(u*(u-1)/2-n)
        if 'diag' in _AGG: TP,FP,FN,TN=2*TP+u,2*FP,2*FN,2*TN
        d=(TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)
        s=0.1+0.9*np.maximum(np.where(d>0,(TP*TN-FP*FN)/np.sqrt(np.maximum(d,1e-9)),0),0)
    else: s=scores_fam(Z,M) if _AGG=='fam' else scores(Z,M)
    if _RND=='floor': return (np.maximum(0,obs-s-1e-6)+np.maximum(0,s-obs-0.05+1e-6)).sum(1)
    if _RND=='ceil': return (np.maximum(0,s-obs-1e-6)+np.maximum(0,obs-0.05-s+1e-6)).sum(1)
    return np.maximum(0,np.abs(s-obs)-tol-1e-6).sum(1)
S8M=[23,33,4,20,30,6,8,16,29,31,10,13];FP8=[15,22];ANC={0:23,1:20,2:16}
def lp_fix(Z,prior):
    v=np.zeros(len(Z))
    if prior.startswith('crp'):
        for e in S8M: v+=Z[:,ix[e]]!=Z[:,ix[ANC[famof[ix[e]]]]]
        for e in FP8:
            f=famof[ix[e]];blk=Z[:,off[f]:off[f]+nf[f]]
            v+=(blk==Z[:,[ix[e]]]).sum(1)-1
    else:
        for e in S8M: v+=Z[:,ix[e]]!=0
        for e in FP8: v+=Z[:,ix[e]]==0
    return -12.0*v
def lp_part(Z,prior):
    if prior.startswith('fix'): return lp_fix(Z,prior[3:])+lp_part(Z,prior[3:])
    if prior.startswith('crp'):
        a=float(prior[3:]);lp=np.zeros(len(Z))
        for f in range(3):
            n=nf[f];sz=(Z[:,off[f]:off[f]+n][:,:,None]==np.arange(n)[None,None,:]).sum(1)
            Kb=(sz>0).sum(1)
            lp+=Kb*np.log(a)+gammaln(np.maximum(sz,1)).sum(1)+gammaln(n-Kb+1)
        return lp
    if prior.startswith('k5'):
        O=(Z==0);v=O[:,[ix[e] for e in K5OUT]].sum(1).astype(float)
        nb=O[:,[ix[e] for e in K5B]].sum(1);na=O[:,[ix[e] for e in K5A]].sum(1)
        v+=np.abs(nb-1)+np.maximum(0,1-na)+np.maximum(0,na-2)
        return -4.0*v+O[:,[ix[e] for e in K5B]]@K5BW+O[:,[ix[e] for e in K5A]]@K5AW+np.log(K5P2/(1-K5P2)/0.4176)*(na==2)
    if prior.startswith('meas'):
        return (Z==0)@MEAS_LOGIT
    if prior.startswith('h'):   # h0.5 : base block + singletons
        r=float(prior[1:]);return (Z==0).sum(1)*np.log(r/(1-r))
# a priori par endpoint tire des mesures sur les poids (passation 1e9a, 06h00 CET), sans les scores publics
MEAS={**{e:0.5 for e in [26,32,17,25,12,5,2,28,7,27,11,14]},**{e:0.15 for e in [3,21,9,19,18,24,1]}}
import os
K5OUT=[5,7,11,27,2,28,26,9,19];K5B=[32,21,3];K5A=[17,25,12,18,24,1,14]
K5BW=np.log(np.array([0.5,0.25,0.25]));K5AW=np.log(np.array([0.2,0.2,0.2,0.12,0.12,0.12,0.04]));K5P2=float(os.environ.get('P2',0.5))
WBS={'1':{**{e:0.75 for e in [32,17,25,12]},**{e:0.6 for e in [7,27,14]},11:0.5,5:0.35,28:0.35,2:0.25,24:0.15,26:0.02,**{e:0.1 for e in [3,21,9,19,18,1]}},
     '2':{**{e:0.75 for e in [32,17,25,12]},**{e:0.6 for e in [7,27,14]},11:0.5,5:0.15,28:0.15,2:0.15,24:0.15,26:0.02,**{e:0.1 for e in [3,21,9,19,18,1]}},
     '3':{27:0.8,7:0.6,14:0.6,11:0.35,**{e:0.35 for e in [17,25,12]},**{e:0.12 for e in [18,24,1]},**{e:0.15 for e in [5,28,2]},26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '4':{27:0.8,7:0.35,14:0.35,11:0.2,**{e:0.35 for e in [17,25,12]},**{e:0.12 for e in [18,24,1]},**{e:0.15 for e in [5,28,2]},26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '5':{27:0.88,7:0.85,14:0.85,11:0.2,**{e:0.35 for e in [17,25,12]},**{e:0.12 for e in [18,24,1]},**{e:0.15 for e in [5,28,2]},26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '6':{27:0.88,7:0.3,14:0.3,11:0.2,**{e:0.35 for e in [17,25,12]},**{e:0.12 for e in [18,24,1]},**{e:0.15 for e in [5,28,2]},26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '7':{27:0.88,7:0.85,14:0.3,11:0.2,**{e:0.35 for e in [17,25,12]},**{e:0.12 for e in [18,24,1]},**{e:0.15 for e in [5,28,2]},26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '8':{27:0.88,7:0.3,14:0.85,11:0.2,**{e:0.35 for e in [17,25,12]},**{e:0.12 for e in [18,24,1]},**{e:0.15 for e in [5,28,2]},26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '9':{**{e:0.95 for e in [27,7,14]},**{e:0.7 for e in [5,28]},11:0.2,**{e:0.35 for e in [17,12]},25:0.02,**{e:0.12 for e in [18,24,1]},2:0.15,26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '10':{**{e:0.95 for e in [27,7,14]},**{e:0.1 for e in [5,28]},11:0.2,**{e:0.35 for e in [17,12]},25:0.02,**{e:0.12 for e in [18,24,1]},2:0.15,26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '11':{**{e:0.05 for e in [27,7,14]},**{e:0.7 for e in [5,28]},11:0.2,**{e:0.35 for e in [17,12]},25:0.02,**{e:0.12 for e in [18,24,1]},2:0.15,26:0.02,3:0.03,21:0.03,9:0.1,19:0.1},
     '12':{**{e:0.05 for e in [27,7,14]},**{e:0.1 for e in [5,28]},11:0.2,**{e:0.35 for e in [17,12]},25:0.02,**{e:0.12 for e in [18,24,1]},2:0.15,26:0.02,3:0.03,21:0.03,9:0.1,19:0.1}}
if os.environ.get('WB'): MEAS=WBS[os.environ['WB']]
_p=np.array([MEAS.get(e,0.5) for e in eps]);MEAS_LOGIT=np.log(_p/(1-_p))
def init(N,prior,rng,q):
    if 'crp' in prior:
        Z=np.stack([rng.integers(0,nf[famof[e]],N) for e in range(33)],1).astype(np.int8)
    else:
        Z=np.where(rng.random((N,33))<0.5,0,local[None,:]+1).astype(np.int8)
    M=rng.random((N,NPAIR))<q
    return Z,M
def run(prior,seed,N=3000,iters=60000,q=0.3,burn=15000,thin=250,out=None):
    rng=np.random.default_rng(seed)
    Z,M=init(N,prior,rng,q);lq=np.log(q/(1-q))
    MASK=os.environ.get('MASK')
    if MASK:
        qu={'both':0.55,'any':0.16}[MASK];lq=np.log(qu/(1-qu));U=rng.random((N,33))<qu
        mk=(lambda U:U[:,pi]&U[:,pj]) if MASK=='both' else (lambda U:U[:,pi]|U[:,pj])
        M=mk(U);SU=[]
    msum=(lambda M,U=None:U.sum(1)) if MASK else (lambda M,U=None:M.sum(1))
    global _CU
    if MASK: _CU=U
    E=energy(Z,M);L=lp_part(Z,prior)+(U.sum(1) if MASK else M.sum(1))*lq
    ar=np.arange(N);SZ=[];SM=[];t0=time.time()
    for it in range(iters):
        tau=max(0.004,0.3*(0.004/0.3)**(it/(burn*0.8)))
        Z2=Z.copy();M2=M.copy()
        mm=rng.random(N)<0.5
        if MASK:
            U2=U.copy();p=rng.integers(0,33,N);U2[ar[mm],p[mm]]^=True;M2=mk(U2)
        else:
            p=rng.integers(0,NPAIR,N);M2[ar[mm],p[mm]]^=True
        e=rng.integers(0,33,N);
        if 'crp' in prior: new=(rng.random(N)*np.array(nf)[famof[e]]).astype(np.int8)
        else: new=np.where(Z[ar,e]==0,local[e]+1,0).astype(np.int8)
        Z2[ar[~mm],e[~mm]]=new[~mm]
        if MASK: _CU=U2
        E2=energy(Z2,M2);L2=lp_part(Z2,prior)+(U2.sum(1) if MASK else M2.sum(1))*lq
        acc=np.log(rng.random(N))<(L2-L)-(E2-E)/tau
        Z[acc]=Z2[acc];M[acc]=M2[acc]
        if MASK: U[acc]=U2[acc]
        E[acc]=E2[acc];L[acc]=L2[acc]
        if it%2000==0: print(prior,seed,it,f'tau={tau:.4f} feas={np.mean(E==0):.3f} Emin={E.min():.3f} Emean={E.mean():.3f} acc={acc.mean():.3f} t={time.time()-t0:.0f}s',flush=True)
        if it>=burn and it%thin==0:
            g=E==0;SZ.append(Z[g].copy());SM.append(np.packbits(M[g],axis=1))
            if MASK: SU.append(U[g].copy())
    SZ=np.concatenate(SZ);SM=np.concatenate(SM)
    if MASK:
        np.savez_compressed(f"umc_{MASK}_{prior}{os.environ.get('WB','')}{os.environ.get('TAG','')}_{seed}.npz",Z=SZ,U=np.concatenate(SU));print('saved',len(SZ));return
    np.savez_compressed(out or (f"samp_fixwb{os.environ['WB']}_{seed}.npz" if os.environ.get('WB') else f'samp_{prior}_{seed}.npz'),Z=SZ,M=SM)
    print('saved',len(SZ))
if __name__=='__main__':
    prior=sys.argv[1];seed=int(sys.argv[2]);it=int(sys.argv[3]) if len(sys.argv)>3 else 60000
    run(prior,seed,iters=it,burn=it//4)
