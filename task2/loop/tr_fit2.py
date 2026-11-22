"""Fit (T,k,p) par endpoint depuis tr2/X.pkl : max de vraisemblance parmi les configs avec <= tol tokens interdits ; gof z, dispersion au niveau réponse."""
import sys, pickle, numpy as np
D = pickle.load(open(sys.argv[1], 'rb')); tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
def st(e, c):
    n, viol, ll, E, V, t1, et1 = D[e]['sum'][c]; A = D[e]['ans'][c]
    d = A[:, 2] - A[:, 3]; zb = d.sum() / (d.std() * np.sqrt(len(d)) + 1e-9)
    return f'T={c[0]:.2f} k={c[1]:3d} p={c[2]:.2f} viol={int(viol):3d} ll/tok={ll/n:.4f} z={(ll-E)/np.sqrt(V):6.1f} zans={zb:5.1f} disp={(d**2).sum()/V:5.2f} top1z={(t1-et1)/np.sqrt(et1*(1-et1/n)):5.1f}'
for e in sorted(D):
    S = D[e]['sum']; n = next(iter(S.values()))[0]; mv = min(S[c][1] for c in S)
    ok = [c for c in S if S[c][1] <= mv + tol * n]
    best = max(ok, key=lambda c: S[c][2]); un = max((c for c in S if c[1] == 0 and c[2] == 1.0), key=lambda c: S[c][2])
    print(f'{e} n={int(n)} ans={len(D[e]["ans"][best])} | plain {st(e, un)} | best {st(e, best)} | dLL={S[best][2]-S[un][2]:.0f}')
