"""Analyse dec_fit.py output: per endpoint, best decoding config under base weights and goodness of fit."""
import sys, numpy as np, collections, math
EPS = 0.01
def interp(A, TS):
    """refine the T grid x5 by linear interpolation of all sums"""
    f = np.linspace(0, len(TS) - 1, (len(TS) - 1) * 5 + 1); i = np.minimum(f.astype(int), len(TS) - 2); w = (f - i)
    sh = (-1,) + (1,) * (A.ndim - 1)
    return A[i] * (1 - w).reshape(sh) + A[i + 1] * w.reshape(sh), np.interp(f, np.arange(len(TS)), TS)
for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True); TS, PS, KS = list(z['TS']), list(z['PS']), list(z['KS'])
    E = collections.defaultdict(lambda: 0)
    for k, S in zip(z['keys'].tolist(), z['S']): E[k[0]] = E[k[0]] + S
    print('==', fn, '| best config under base weights (mixture: 1% of tokens may fall outside the kept set) ; dLL vs best pure-T ; GOF z at best config: hi-vs-lo entropy contrast, top-1 excess | same for pure T')
    for e in sorted(E):
        A, T2 = interp(E[e], TS)
        sc = A[..., 1] + A[..., 0] * math.log(1 - EPS) + A[..., 2] * math.log(EPS) + A[..., 12]
        sc[:, 0, 0] = A[:, 0, 0, 1]
        def gof(s):
            r_lo, v_lo, r_hi, v_hi = s[3] - s[4], s[5], s[6] - s[7], s[8]
            c = (r_hi / max(v_hi, 1e-9) - r_lo / max(v_lo, 1e-9)) / math.sqrt(1 / max(v_hi, 1e-9) + 1 / max(v_lo, 1e-9))
            return c, (s[9] - s[10]) / math.sqrt(max(s[11], 1e-9)), (s[3] + s[6] - s[4] - s[7]) / math.sqrt(v_lo + v_hi)
        b = np.unravel_index(np.argmax(sc), sc.shape); t0 = int(np.argmax(sc[:, 0, 0]))
        gb, g0 = gof(A[b]), gof(A[t0, 0, 0])
        n = int(A[0, 0, 0, 0])
        print(f"{e} tok={n:5d} | best T={T2[b[0]]:.2f} p={PS[b[1]]:.2f} k={int(KS[b[2]]):3d} out={int(round(A[b][2])):2d} dLL={sc[b]-sc[t0,0,0]:+7.1f} zc={gb[0]:+5.1f} z1={gb[1]:+5.1f} | pureT={T2[t0]:.2f} zc={g0[0]:+5.1f} z1={g0[1]:+5.1f}")
