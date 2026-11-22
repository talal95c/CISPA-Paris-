"""Temperature heterogeneity: T fitted on confident positions (p1>=0.6 at T=1) minus T on the others, with bootstrap SE over answers.
Pure temperature change => dT ~ 0; changed weights / hidden context => dT > 0. Usage: wb_dt.py base_names"""
import sys, numpy as np
from wb_fit import load, GRID, kind
from wb_feat import Tfit
rng = np.random.default_rng(0)
def parts(D, k):
    zx = np.array(D[k]['zx']); lz = np.array([D[k]['g'][str(T)]['logZ'] for T in GRID]).T; hi = np.array(D[k]['g']['1.0']['p1']) >= 0.6
    return zx, lz, hi
def dT(P):
    zx = np.concatenate([p[0] for p in P]); lz = np.concatenate([p[1] for p in P]); hi = np.concatenate([p[2] for p in P])
    return Tfit(zx, lz), Tfit(zx[hi], lz[hi]) - Tfit(zx[~hi], lz[~hi])
if __name__ == '__main__':
    D = load(sys.argv[1])
    for e in sorted({k[0] for k in D}, key=lambda e: (kind.get(e, '') if e.startswith('ref') else '', e)):
        P = [parts(D, k) for k in D if k[0] == e]; T, d = dT(P)
        B = np.array([dT([P[i] for i in rng.integers(0, len(P), len(P))]) for _ in range(200)])
        print(f'{e} {kind.get(e, "")[:14]:<14} n={len(P):3d} T={T:.3f}±{B[:,0].std():.3f} dT={d:+.3f}±{B[:,1].std():.3f} z={d/B[:,1].std():5.1f}')
