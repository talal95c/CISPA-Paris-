"""Analyse token-count residuals (resid_wb.py output). Usage: resid_an.py wb/res_X.npz [id-prefix-filter]
Split-half design: answers are split into two prompt-disjoint halves; residual directions are compared across halves."""
import json, sys, numpy as np
from wb_fit import kind

def loadres(path, keep=lambda k: True):
    z = np.load(path); keys = [tuple(k) for k in json.loads(str(z['keys']))]; obs = json.loads(str(z['obs']))
    E = z['E'].astype(np.float32); V = z['V'].astype(np.float32)
    O = np.zeros_like(E)
    for i, o in enumerate(obs): np.add.at(O[i], o, 1)
    idx = [i for i, k in enumerate(keys) if keep(k)]
    return [keys[i] for i in idx], O[idx], E[idx], V[idx]

def agg(keys, O, E, V, sel):
    """per endpoint: summed residual and variance over answers selected by sel(key)"""
    out = {}
    for e in sorted({k[0] for k in keys}):
        m = np.array([k[0] == e and sel(k) for k in keys])
        if m.any(): out[e] = ((O[m] - E[m]).sum(0), V[m].sum(0), int(O[m].sum()))
    return out

def zscore(d, R, Vv):
    """score-test z of residual R along direction d (multinomial variance, tokens treated as independent)"""
    return float(d @ R / np.sqrt((d * d) @ Vv + 1e-9))

if __name__ == '__main__':
    pref = sys.argv[2].split(',') if len(sys.argv) > 2 else None
    keys, O, E, V = loadres(sys.argv[1], (lambda k: any(k[1].startswith(p) for p in pref)) if pref else (lambda k: True))
    ids = sorted({k[1] for k in keys}); h = {i: n % 2 for n, i in enumerate(ids)}
    H = [agg(keys, O, E, V, lambda k, s=s: h[k[1]] == s) for s in (0, 1)]
    eps = sorted(H[0], key=lambda e: (kind.get(e, ''), e))
    print('rows: direction from half 0 (standardised residual), cols: z of half-1 residual along it')
    print(' ' * 24 + ' '.join(f'{e[-5:]:>6}' for e in eps))
    for e in eps:
        R, Vv, n = H[0][e]; d = R / np.sqrt(Vv + 0.5)
        print(f'{e} {kind.get(e, "")[:15]:<16}' + ' '.join(f'{zscore(d, *H[1][f][:2]):6.1f}' for f in eps))
