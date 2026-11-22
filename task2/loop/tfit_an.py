import sys, pickle, numpy as np, collections
R = pickle.load(open(sys.argv[1], 'rb')); TS = np.round(np.arange(0.62, 0.781, 0.02), 2); rng = np.random.default_rng(0)
by = collections.defaultdict(list)
for d in R: by[d['ep']].append(d)
def That(A):
    L = np.array([sum(d[T]['ll'].sum() for d in A) for T in TS]); i = int(L.argmax())
    if 0 < i < len(TS) - 1:
        a, b, c = L[i - 1], L[i], L[i + 1]; return TS[i] + 0.02 * 0.5 * (a - c) / (a - 2 * b + c)
    return TS[i]
for e, A in sorted(by.items()):
    t = That(A); B = [That([A[j] for j in rng.integers(0, len(A), len(A))]) for _ in range(200)]
    T = TS[np.abs(TS - t).argmin()]
    ll = np.concatenate([d[T]['ll'] for d in A]); E = np.concatenate([d[T]['E'] for d in A]); V = np.concatenate([d[T]['V'] for d in A])
    p1 = np.concatenate([d[T]['p1'] for d in A]); rk = np.concatenate([d['rank'] for d in A]); n = len(ll)
    da = np.array([(d[T]['ll'] - d[T]['E']).sum() for d in A]); va = np.array([d[T]['V'].sum() for d in A])
    med = np.median(-E)
    def Tsub(f): return TS[int(np.argmax([sum(d[TT]['ll'][f(d)].sum() for d in A) for TT in TS]))]
    Thi = Tsub(lambda d: -d[T]['E'] > med); Tlo = Tsub(lambda d: -d[T]['E'] <= med)
    eo = sum(d['full'] for d in A); ee = sum(d[T]['peos'].sum() for d in A)
    print(f"{e} ans={len(A)} tok={n} T={t:.3f}±{np.std(B):.3f} z={(ll-E).sum()/np.sqrt(V.sum()):5.1f} disp={((ll-E)**2).sum()/V.sum():.2f} dispAns={(da**2/va).mean():.2f} top1z={((rk==0).sum()-p1.sum())/np.sqrt((p1*(1-p1)).sum()):5.1f} r>=20={np.mean(rk>=20)*100:.2f}% r>=100={np.mean(rk>=100)*100:.2f}% Thi={Thi:.2f} Tlo={Tlo:.2f} eos={eo}/{ee:.1f} len={n/len(A):.0f}")
