import sys,numpy as np,collections
rng=np.random.default_rng(0)
pre=sys.argv[1];vars_=sys.argv[2].split(',')
B=np.load(f'{pre}_base.npz');TS=[0.65,0.7,0.75,0.8,1.0]
eps=sorted({k.split('|')[0] for k in B.files})
for e in eps:
    ks=[k for k in B.files if k.startswith(e)]
    tot=np.array([sum(B[k][i].sum() for k in ks) for i in range(5)]);n=sum(B[k].shape[1] for k in ks);ti=int(tot.argmax())
    rk=np.concatenate([B[k][5] for k in ks])
    s=f'{e} ans={len(ks)} tok={n} bestT={TS[ti]} ll/tok={tot[ti]/n:.4f} rank>=50: {(rk>=50).sum()} top1 {np.mean(rk==0):.3f} |'
    for v in vars_:
        try: V=np.load(f'{pre}_{v}.npz')
        except Exception: continue
        kk=[k for k in ks if k in V.files]
        if not kk: continue
        d=np.array([(B[k][ti]-V[k][ti]).sum() for k in kk]);nn=np.array([B[k].shape[1] for k in kk])
        bs=[d[i].sum()/nn[i].sum()*1000 for i in (rng.integers(0,len(kk),len(kk)) for _ in range(500))]
        s+=f' {v}: {d.sum()/nn.sum()*1000:+.2f}±{np.std(bs):.2f}/ktok (n={len(kk)})'
    print(s)
