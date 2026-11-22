"""Bayesian design of the next leaderboard probe. Truth model: one base-weights group per family (+ singletons), core members fixed by
white-box evidence, flat prior on the rest. Posterior from all past public scores (gaussian error sigma on the score, all-pairs approximation).
For each candidate probe (S5 core + a subset of the uncertain endpoints) print the expected score, the predictive score histogram and the
expected information gain. Usage: lb_design.py [sigma] [extra lookalike candidates e.g. "26 13"]"""
import sys, itertools, numpy as np
_src = open('lb_post.py').read(); exec(_src[:_src.index('G = [[')])           # FAM, SUBS, MUST, OUT, NP
G = [[set(g.split()) for g in s[2]] for s in SUBS]; obs = np.array([s[1] for s in SUBS])
npred = np.array([sum(len(g) * (len(g) - 1) // 2 for g in gs) for gs in G])
c2 = lambda n: n * (n - 1) // 2
H = {}
for f in FAM:
    E = FAM[f]; L = []
    for m in range(1 << len(E)):
        S = {E[i] for i in range(len(E)) if m >> i & 1}
        if not MUST[f] <= S or S & OUT: continue
        L.append(S)
    H[f] = L
def score(P, tp, n):
    fp = n - tp; fn = P - tp; tn = NP - tp - fp - fn
    d = np.sqrt(np.maximum((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1e-9))
    return .1 + .9 * np.maximum(0, (tp * tn - fp * fn) / d)
Pf = {f: np.array([c2(len(S)) for S in H[f]]) for f in FAM}
TPf = {f: np.array([[sum(c2(len(S & g)) for g in gs) for gs in G] for S in H[f]]) for f in FAM}       # hyp x subs
P = Pf['A'][:, None, None] + Pf['B'][None, :, None] + Pf['C'][None, None, :]
ll = np.zeros(P.shape)
for j in range(len(SUBS)):
    tp = TPf['A'][:, j][:, None, None] + TPf['B'][:, j][None, :, None] + TPf['C'][:, j][None, None, :]
    ll -= (score(P, tp, npred[j]) - obs[j]) ** 2 / (2 * sig ** 2)
W = np.exp(ll - ll.max()); W /= W.sum()
for f, ax in zip('ABC', ((1, 2), (0, 2), (0, 1))):
    w = W.sum(ax); print(f, ' '.join(f'{e}:{sum(w[i] for i, S in enumerate(H[f]) if e in S):.2f}' for e in FAM[f]))
print('P posterior mean %.1f' % (W * P).sum())
if __name__ == '__main__':
    U = '14 07 02 28 11 05 27'.split() + (sys.argv[2].split() if len(sys.argv) > 2 else [])
    fam = {e: f for f, v in FAM.items() for e in v}; res = []
    bins = np.arange(.075, 1.03, .05)
    for r in range(len(U) + 1):
        for add in itertools.combinations(U, r):
            g = {f: MUST[f] | {e for e in add if fam[e] == f} for f in FAM}
            n = sum(c2(len(g[f])) for f in FAM)
            tp = (np.array([c2(len(S & g['A'])) for S in H['A']])[:, None, None] + np.array([c2(len(S & g['B'])) for S in H['B']])[None, :, None]
                  + np.array([c2(len(S & g['C'])) for S in H['C']])[None, None, :])
            s = score(P, tp, n); h = np.histogram(s.ravel(), bins=bins, weights=W.ravel())[0]
            ent = -(h[h > 0] * np.log2(h[h > 0])).sum()
            res.append((ent, (W * s).sum(), add, h))
    res.sort(key=lambda t: -t[0])
    for ent, es, add, h in res[:int(sys.argv[3]) if len(sys.argv) > 3 else 14]:
        print(f'info={ent:.2f} bits  E[score]={es:.2f}  +{" ".join(add):26s} ' + ' '.join(f'{b + .025:.2f}:{x:.2f}' for b, x in zip(bins, h) if x > .03))
    res.sort(key=lambda t: -t[1]); print('best expected score:')
    for ent, es, add, h in res[:5]:
        print(f'info={ent:.2f} bits  E[score]={es:.2f}  +{" ".join(add):26s} ' + ' '.join(f'{b + .025:.2f}:{x:.2f}' for b, x in zip(bins, h) if x > .03))
