import sys, pickle, numpy as np, collections
R = pickle.load(open(sys.argv[1], 'rb')); by = collections.defaultdict(list); rng = np.random.default_rng(0)
TS = np.round(np.arange(0.5, 1.301, 0.05), 2); i7 = int(np.abs(TS - 0.7).argmin())
for d in R: by[d['ep']].append(d)
for e, A in sorted(by.items()):
    def z1(A):
        p1 = np.concatenate([d['p1'] for d in A]); rk = np.concatenate([d['rank'] for d in A])
        return ((rk == 0).sum() - p1.sum()) / np.sqrt((p1 * (1 - p1)).sum())
    ll = np.array([d['ll'][i7].sum() for d in A]); n = np.array([len(d['rank']) for d in A])
    h = [z1(A[k::2]) for k in (0, 1)]
    print(f"{e} ans={len(A)} top1z@.7={z1(A):+.2f} halves {h[0]:+.2f}/{h[1]:+.2f}  ll/tok@.7={ll.sum()/n.sum():.4f}")
