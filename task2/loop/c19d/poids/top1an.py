import sys,numpy as np,collections
E=collections.defaultdict(list)
for f in sys.argv[1:]:
    d=np.load(f);TS=d['TS'];n=len(TS)
    for k in d.files:
        if k!='TS': E[k.split('|')[0]].append(d[k])
i7=int(np.abs(TS-0.70).argmin());rng=np.random.default_rng(0)
def stat(v):
    L=sum(a[:n].sum(1) for a in v);T=TS[L.argmax()]
    I=np.concatenate([a[-1] for a in v]);p=np.concatenate([a[n+i7] for a in v])
    return T,(I-p).sum()/np.sqrt((p*(1-p)).sum()),len(I)
print('ep    n   tok   T*    z_top1(T=0.7)  [bootstrap sd T, sd z]')
for e in sorted(E):
    v=E[e];T,z,nt=stat(v);B=[stat([v[j] for j in rng.integers(0,len(v),len(v))]) for _ in range(200)]
    print(e,'%3d %5d  %.2f   %+.2f     [%.3f, %.2f]'%(len(v),nt,T,z,np.std([b[0] for b in B]),np.std([b[1] for b in B])))
