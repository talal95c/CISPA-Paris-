"""Stop-hazard test under a known base checkpoint.
At every position the base model (at the endpoint's fitted temperature) gives a probability of ending the turn.
An endpoint that samples from the base weights stops as often as predicted; a fine-tuned one (LoRA) does not.
Usage: wb_eos.py name [id-prefix,...]"""
import json, sys, numpy as np
from wb_fit import load, GRID, kind
from wb_feat import stack, Tfit
G = np.array(GRID); U = 1 / G

def interp(x, key, T):
    a = np.array([x['g'][str(t)][key] for t in GRID])           # T x positions
    j = int(np.clip(np.searchsorted(-U, -1 / T) - 1, 0, len(G) - 2)); w = (1 / T - U[j]) / (U[j + 1] - U[j])
    return (1 - w) * a[j] + w * a[j + 1]

def caps(D):
    """per prompt id: the token cap, inferred as the largest answer length seen for that prompt"""
    c = {}
    for k, x in D.items(): c[k[1]] = max(c.get(k[1], 0), x['n'])
    return c

def hazard(D, e, T, cap, keys=None):
    exp = var = stops = early = n = 0
    for k, x in D.items():
        if k[0] != e or (keys is not None and k not in keys): continue
        p = np.exp(interp(x, 'eos', T)); stopped = x['n'] < min(cap[k[1]], 256)
        early += p[:-1].sum()                                    # stops the base model expected before the observed end
        q = p if stopped else p[:-1]
        exp += q.sum(); var += (q * (1 - q)).sum(); stops += stopped; n += 1
    return dict(n=n, stops=stops, exp=exp, early=early, z=(stops - exp) / np.sqrt(var + 1e-9))

if __name__ == '__main__':
    pref = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    D = load(sys.argv[1], (lambda k: any(k[1].startswith(p) for p in pref)) if pref else (lambda k: True))
    cap = caps(D); ids = sorted(cap); h = {i: n % 2 for n, i in enumerate(ids)}
    for e in sorted({k[0] for k in D}, key=lambda e: (kind.get(e, ''), e)):
        ks = [k for k in D if k[0] == e]; T = Tfit(*stack(D, ks))
        s = f'{e} {kind.get(e, "")[:15]:<16} T={T:.3f}'
        for name, sel in (('all', None), ('h0', {k for k in ks if h[k[1]] == 0}), ('h1', {k for k in ks if h[k[1]] == 1})):
            r = hazard(D, e, T, cap, sel)
            s += f' | {name} n={r["n"]:3d} stops={r["stops"]:3d} exp={r["exp"]:6.1f} early={r["early"]:6.1f} z={r["z"]:6.1f}'
        print(s)
