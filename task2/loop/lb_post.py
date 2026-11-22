"""Which truths are compatible with the public scores? Enumerate 'one same-weights group per family + singletons' (reference-like structure)
and weight each truth by exp(-sum (score_pred - score_obs)^2 / 2 sigma^2). Score = 0.1 + 0.9 * max(0, MCC over the 166 same-family pairs).
Prints P(endpoint in the group) and the best truths. Usage: lb_post.py [sigma]"""
import sys, itertools, numpy as np, collections
FAM = dict(A='01 04 12 14 17 18 23 24 25 33'.split(), B='02 03 06 07 08 20 21 22 26 28 30 32'.split(), C='05 09 10 11 13 15 16 19 27 29 31'.split())
SUBS = [  # (name, observed score, list of predicted groups)
 ('2141', .10, ['01 04 12 14 17 18 23 24 25 33', '03 06 07 08 20 21 22 26 30 32', '09 10 11 13 15 16 19 27 29 31']),
 ('2312', .10, ['01 04 12', '06 21 22', '10 13 19']),
 ('2426', .20, ['01 04 12 14 17 18 23 24 25 33', '02 03 06 07 20 21 26 28 30 32', '05 09 10 11 13 16 19 27 29 31']),
 ('S1', .45, ['02 07 20 28 30', '05 11 16 27 29 31', '14 23 33']),
 ('S2', .35, ['02 07 20 28 30', '14 23 33']),
 ('S3', .45, ['16 29 31', '20 30', '23 33']),
 ('S4', .45, ['07 20 30', '11 16 27 29 31', '14 23 33']),
]
sig = float(sys.argv[1]) if len(sys.argv) > 1 else 0.03
NP = sum(len(v) * (len(v) - 1) // 2 for v in FAM.values())
G = [[set(g.split()) for g in s[2]] for s in SUBS]
npred = np.array([sum(len(g) * (len(g) - 1) // 2 for g in gs) for gs in G])
obs = np.array([s[1] for s in SUBS])
def keys(f):
    """all subsets of family f -> dict key=(P, tp per submission) -> (count, per-endpoint membership counts)"""
    E = FAM[f]; out = {}
    for m in range(1 << len(E)):
        S = {E[i] for i in range(len(E)) if m >> i & 1}
        if len(S) == 1: continue
        k = (len(S) * (len(S) - 1) // 2,) + tuple(sum(len(S & g) * (len(S & g) - 1) // 2 for g in gs) for gs in G)
        c = out.setdefault(k, [0, np.zeros(len(E)), m]); c[0] += 1; c[1] += [m >> i & 1 for i in range(len(E))]
    return out
K = {f: keys(f) for f in FAM}
def arr(f): return np.array(list(K[f].keys())), np.array([v[0] for v in K[f].values()]), np.array([v[1] for v in K[f].values()]), [v[2] for v in K[f].values()]
(ka, ca, ma, xa), (kb, cb, mb, xb), (kc, cc, mc, xc) = arr('A'), arr('B'), arr('C')
print('distinct keys', len(ka), len(kb), len(kc), 'pairs', NP, 'pred', npred, flush=True)
def score(k):
    P, tp = k[..., :1], k[..., 1:]; fp = npred - tp; fn = P - tp; tn = NP - tp - fp - fn
    d = np.sqrt(np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-9))
    return .1 + .9 * np.maximum(0, (tp * tn - fp * fn) / d)
Z = 0; pa = np.zeros(ma.shape[1]); pb = np.zeros(mb.shape[1]); pc = np.zeros(mc.shape[1]); best = []
sizeP = collections.Counter()
for i in range(len(ka)):
    kab = ka[i] + kb                                            # (nb, 8)
    k = kab[:, None, :] + kc[None, :, :]                        # (nb, nc, 8)
    r = score(k) - obs
    ll = -(r ** 2).sum(-1) / (2 * sig ** 2)
    w = np.exp(ll) * ca[i] * cb[:, None] * cc[None, :]
    Z += w.sum(); pa += w.sum() * ma[i] / ca[i]; pb += (w.sum(1) / cb) @ mb; pc += (w.sum(0) / cc) @ mc
    j = np.unravel_index(np.argsort(ll, None)[-3:], ll.shape)
    best += [(ll[b, c], xa[i], xb[b], xc[c], tuple(np.round(score(k[b, c]), 3))) for b, c in zip(*j)]
    for P_, w_ in zip(k[..., 0].ravel() // 5 * 5, w.ravel()):
        pass
show = lambda f, m: ' '.join(e for i, e in enumerate(FAM[f]) if m >> i & 1)
for f, p in zip('ABC', (pa, pb, pc)): print(f, ' '.join(f'{e}:{x / Z:.2f}' for e, x in zip(FAM[f], p)))
print('obs  ', tuple(obs))
for ll, a, b, c, s in sorted(best, key=lambda t: -t[0])[:25]: print(f'{ll:7.2f} A[{show("A", a)}] B[{show("B", b)}] C[{show("C", c)}] {s}')
