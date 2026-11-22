"""Truncated-sampling test: observed tokens that top-k / top-p would forbid. Per endpoint: T-hat, tokens, count rank>=50, rank>=20,
count with p_T(x)<1e-2 / 3e-3 / 1e-3 (p at fitted T, top-4096 normalisation). Usage: wb_trunc.py base_names"""
import sys, numpy as np
from wb_fit import load, GRID, kind
from wb_feat import stack, Tfit
D = load(sys.argv[1])
for e in sorted({k[0] for k in D}):
    ks = [k for k in D if k[0] == e]; T = Tfit(*stack(D, ks)); g = [t for t in GRID]
    r = np.concatenate([D[k]['rank'] for k in ks]); zx = np.concatenate([D[k]['zx'] for k in ks])
    lo, hi = max(t for t in g if t <= T) if T >= g[0] else g[0], min(t for t in g if t >= T) if T <= g[-1] else g[-1]
    w = 0 if hi == lo else (T - lo) / (hi - lo)
    lz = (1 - w) * np.concatenate([D[k]['g'][str(lo)]['logZ'] for k in ks]) + w * np.concatenate([D[k]['g'][str(hi)]['logZ'] for k in ks])
    p = np.exp(zx / T - lz)
    print(f'{e} {kind.get(e, "")[:10]:<10} T={T:.3f} tok={len(r):5d} r>=50:{(r >= 50).sum():3d} r>=20:{(r >= 20).sum():3d} p<.01:{(p < .01).sum():3d} p<.003:{(p < .003).sum():3d} p<.001:{(p < .001).sum():3d} minp={p.min():.5f}')
