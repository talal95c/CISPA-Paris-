"""Candidate-free weight fingerprint analysis on uni/NAME.pkl (see score_uni.py).
r_e = O - E(T-hat) with ALL temperature directions (differences between grid temperatures) projected out (weighted LS);
z_ef = <r_e, r_f>_w / sqrt(<V_e, V_f>_w2) ~ N(0,1) if independent residuals.
Usage: uni_sim.py [--w raw|chi] [--common] uni/name [uni/name2 ...]"""
import sys, pickle, numpy as np
from wb_fit import kind
TS = [0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
args = sys.argv[1:]; W = 'raw'; COMMON = False
if args[0] == '--w': W = args[1]; args = args[2:]
if args[0] == '--common': COMMON = True; args = args[1:]
def resid(O, E, V):
    chi = [((O - E[i]) ** 2 / (V[i] + 1e-3)).sum() for i in range(len(TS))]
    i = int(np.argmin(chi)); w = 1 / (V[i] + 1e-3)
    keep = V[i] > 1e-6                                   # tokens that can plausibly occur
    D = np.stack([E[j] - E[i] for j in range(len(TS)) if j != i], 1)      # nv x 5 temperature directions
    sw = np.sqrt(w)[:, None]
    coef, *_ = np.linalg.lstsq(D * sw, (O - E[i]) * sw[:, 0], rcond=1e-8)
    r = (O - E[i]) - D @ coef
    That = TS[i] + sum(c * (TS[j] - TS[i]) for c, j in zip(coef, [j for j in range(len(TS)) if j != i]))
    return r * keep, V[i] * keep, That
def load(files):
    S = {}
    for f in files:
        P = pickle.load(open(f + '.pkl', 'rb'))
        for e in P['O']: S[e] = P
    R, VV, T = {}, {}, {}
    for e in sorted(S):
        P = S[e]
        R[e, 2], VV[e, 2], T[e] = resid(P['O'][e].sum(0), P['E'][e].sum(0), P['V'][e].sum(0))
        for h in (0, 1): R[e, h], VV[e, h], _ = resid(P['O'][e][h], P['E'][e][h], P['V'][e][h])
    return S, R, VV, T
def zf(R, VV, a, b, W='raw'):
    if W == 'raw': return (R[a] * R[b]).sum() / np.sqrt((VV[a] * VV[b]).sum())
    va, vb = VV[a] + 0.5, VV[b] + 0.5                    # chi-weighted: each token standardized (0.5 pseudo-variance for rare tokens)
    return (R[a] * R[b] / np.sqrt(va * vb)).sum() / np.sqrt((VV[a] * VV[b] / (va * vb)).sum())
if __name__ == '__main__':
    S, R, VV, T = load(args); eps = sorted(S)
    if COMMON:                                           # remove the family-wide mean residual (shared misfit of the checkpoint/template)
        for h in (0, 1, 2):
            m = np.mean([R[e, h] / max(1, sum(S[e]['n'][e]) if h == 2 else S[e]['n'][e][h]) for e in eps], 0)
            for e in eps: R[e, h] = R[e, h] - m * (sum(S[e]['n'][e]) if h == 2 else S[e]['n'][e][h])
    z = lambda a, b: zf(R, VV, a, b, W)
    print('self-consistency between halves (z), T-hat, tokens')
    for e in eps: print(f'  {e} {kind.get(e, "")[:14]:<14} self={z((e, 0), (e, 1)):6.1f}  T={T[e]:.3f}  n={sum(S[e]["n"][e])}')
    print('cross z (all answers)')
    print('       ' + ' '.join(e[-2:].rjust(5) for e in eps))
    for a in eps: print(f'  {a[-5:]} ' + ' '.join(('  .  ' if a == b else f'{z((a, 2), (b, 2)):5.1f}') for b in eps))
