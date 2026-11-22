import sys, pickle, numpy as np
TS = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1]
R = pickle.load(open(sys.argv[1], 'rb')); eps = sorted({d['ep'] for d in R})
names = list(R[0]['out'])
for e in eps:
    A = [d for d in R if d['ep'] == e]; L = np.array([[d['ll'][T] for T in TS] for d in A]).sum(0); T = TS[int(L.argmax())]; n = sum(d['n'] for d in A)
    rk = np.concatenate([d['rank'] for d in A])
    s = f"{e} ans={len(A)} tok={n} T={T} r>=20:{(rk>=20).sum()} r>=50:{(rk>=50).sum()} r>=100:{(rk>=100).sum()} |"
    for name in names:
        o = sum(d['out'][name].sum() for d in A); ex = sum(d['mass'][name][T].sum() for d in A)
        s += f" {name} {o}/{ex:.1f};"
    print(s)
