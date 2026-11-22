"""Best decoding config per endpoint from tr/NAME.pkl: among configs with <= tol forbidden tokens, maximise the log-likelihood.
Prints the fit: ll per token, goodness-of-fit z = (ll - E)/sqrt(V) (0 if the answers come from the base weights with this decoding),
bootstrap SE over answers, top-1 excess z. Also the best untruncated config. Usage: tr_fit.py tr/name.pkl [tol_fraction=0.002]"""
import sys, pickle, numpy as np
rng = np.random.default_rng(0)
D = pickle.load(open(sys.argv[1], 'rb')); tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.002
def line(e, c, tag):
    n, viol, ll, E, V, t1, et1 = D[e]['sum'][c]; A = D[e]['ans'][c]
    z = (ll - E) / np.sqrt(V); d = (A[:, 2] - A[:, 3]); zb = d.sum() / (d.std() * np.sqrt(len(d)) + 1e-9)
    return f'  {tag:6s} T={c[0]:.1f} k={c[1]:3d} p={c[2]:.2f} viol={int(viol):3d}/{int(n)} ll/tok={ll / n:.4f} gof_z={z:6.1f} (answer-level {zb:5.1f}) disp={(d ** 2).sum() / V:5.2f} top1_z={(t1 - et1) / np.sqrt(et1 * (1 - et1 / n)):5.1f}'
for e in D:
    S = D[e]['sum']; n = next(iter(S.values()))[0]
    ok = [c for c in S if S[c][1] <= tol * n]
    best = max(ok, key=lambda c: S[c][2] + (-1e9 if False else 0)); un = max((c for c in S if c[1] == 0 and c[2] == 1.0), key=lambda c: S[c][2])
    print(e, f'answers={len(D[e]["ans"][best])}'); print(line(e, un, 'plain')); print(line(e, best, 'best'))
    for c in sorted(ok, key=lambda c: -S[c][2])[1:4]: print(line(e, c, 'next'))
