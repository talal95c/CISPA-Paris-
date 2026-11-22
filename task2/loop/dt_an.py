import sys, pickle, numpy as np, collections
TS = np.round(np.arange(0.5, 1.301, 0.05), 2); rng = np.random.default_rng(0)
R = pickle.load(open(sys.argv[1], 'rb')); by = collections.defaultdict(list)
for d in R: by[d['ep']].append(d)
def tf(L):
    i = int(L.argmax())
    if 0 < i < len(TS) - 1:
        a, b, c = L[i - 1], L[i], L[i + 1]; return TS[i] + 0.05 * 0.5 * (a - c) / (a - 2 * b + c)
    return TS[i]
def stats(A):
    al = sum(d['ll'].sum(1) for d in A)
    hi = sum(d['ll'][:, d['p1'] < 0.6].sum(1) for d in A); lo = sum(d['ll'][:, d['p1'] >= 0.6].sum(1) for d in A)
    return tf(al), tf(hi), tf(lo)
for e, A in sorted(by.items()):
    t, h, l = stats(A); B = np.array([stats([A[j] for j in rng.integers(0, len(A), len(A))]) for _ in range(300)])
    rk = np.concatenate([d['rank'] for d in A]); n = len(rk)
    print(f"{e} ans={len(A)} tok={n} T={t:.3f}±{B[:,0].std():.3f} Tflat(p1<.6)={h:.3f} Tpeaked={l:.3f} dT(peaked-flat)={l-h:+.3f}±{(B[:,2]-B[:,1]).std():.3f} r>=40:{(rk>=40).sum()} r>=100:{(rk>=100).mean()*100:.2f}%")
