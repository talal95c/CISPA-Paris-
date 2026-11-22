"""Analyse tail_fit.py: per endpoint, pure-T MLE under base weights, then z-scores of (observed - expected) counts of tokens
of rank >= K / outside the top-p nucleus, entropy contrast and top-1 excess. usage: tail_an.py [--min N] file.npz ..."""
import sys, numpy as np, math
args = sys.argv[1:]; MIN = 0
if args[0] == '--min': MIN = int(args[1]); args = args[2:]
def at_T(A, TS):
    r = A[:, 1] - A[:, 2]
    for i in range(len(TS) - 1):
        if (r[i] <= 0 < r[i + 1]) or (r[i] >= 0 > r[i + 1]):
            w = r[i] / (r[i] - r[i + 1]); return TS[i] * (1 - w) + TS[i + 1] * w, A[i] * (1 - w) + A[i + 1] * w
    i = int(np.argmin(np.abs(r))); return TS[i], A[i]
for fn in args:
    z = np.load(fn, allow_pickle=True); TS, KS, PS = list(z['TS']), list(z['KS']), list(z['PS']); keys = z['keys'].tolist(); S = z['S']
    names = [f'k{int(k)}' for k in KS] + [f'p{p}' for p in PS]
    print('==', fn, f'(answers with >= {MIN} tokens) | z = (obs-exp)/sd under base at fitted T ; zc = high-entropy residual ; z1 = top-1 excess')
    for e in sorted({k[0] for k in keys}):
        idx = [i for i, k in enumerate(keys) if k[0] == e and S[i][0, 0] >= MIN]
        if not idx: continue
        T, s = at_T(S[idx].sum(0), TS)
        out = f"{e} ans={len(idx):3d} tok={int(s[0]):5d} T={T:.3f} zc={(s[4]-s[5])/math.sqrt(max(s[6],1e-9)):+5.1f}"
        j = 7
        for nm in names:
            out += f" {nm}={(s[j]-s[j+1])/math.sqrt(max(s[j+2],1e-9)):+5.1f}"; j += 3
        out += f" z1={(s[j]-s[j+1])/math.sqrt(max(s[j+2],1e-9)):+5.1f} | k20 {int(s[13])}/{s[14]:.0f} k50 {int(s[16])}/{s[17]:.0f}"
        print(out)
