"""Per-endpoint white-box features under a known base checkpoint (and its simulated fp8 version).
Usage: wb_feat.py base_name [fp8_name]"""
import json, sys, numpy as np
from wb_fit import load, GRID, kind
G = np.array(GRID)

def stack(D, ks):
    zx = np.concatenate([D[k]['zx'] for k in ks])
    logZ = np.concatenate([np.array([D[k]['g'][str(T)]['logZ'] for T in GRID]).T for k in ks])
    return zx, logZ

def Tfit(zx, logZ):
    L = (zx[:, None] / G[None, :] - logZ).sum(0); i = int(L.argmax())
    if 0 < i < len(G) - 1:
        x = 1 / G[i - 1:i + 2]; c = np.polyfit(x, L[i - 1:i + 2], 2)
        if c[0] < 0: return float(1 / (-c[1] / (2 * c[0])))
    return float(G[i])

def llT(x, T):
    """per-token log p_T of one answer, by linear interpolation of logZ on the grid in 1/T"""
    zx = np.array(x['zx']); lz = np.array([x['g'][str(t)]['logZ'] for t in GRID])   # T x tok
    u = 1 / G; j = np.clip(np.searchsorted(-u, -1 / T) - 1, 0, len(G) - 2)
    w = (1 / T - u[j]) / (u[j + 1] - u[j])
    return zx / T - ((1 - w) * lz[j] + w * lz[j + 1])

def features(D, F=None, eps=None):
    out = {}
    for e in eps or sorted({k[0] for k in D}):
        ks = [k for k in D if k[0] == e and (F is None or k in F)]
        zx, logZ = stack(D, ks)
        p1 = np.concatenate([D[k]['g']['1.0']['p1'] for k in ks]); rank = np.concatenate([D[k]['rank'] for k in ks])
        T = Tfit(zx, logZ); hi = p1 >= 0.6
        f = dict(n=len(ks), tok=len(zx), T=T, Thi=Tfit(zx[hi], logZ[hi]), Tlo=Tfit(zx[~hi], logZ[~hi]),
                 tail=float(np.mean(rank >= 100)), top1=float(np.mean(rank == 0)))
        if F is not None:
            zf, lf = stack(F, ks); Tf = Tfit(zf, lf)
            d = np.array([llT(D[k], T).sum() - llT(F[k], Tf).sum() for k in ks])
            f.update(Tf=Tf, llr=float(1000 * d.sum() / len(zx)), z=float(d.mean() / (d.std() + 1e-12) * np.sqrt(len(d))))
        out[e] = f
    return out

if __name__ == '__main__':
    D = load(sys.argv[1]); F = load(sys.argv[2]) if len(sys.argv) > 2 else None
    Fe = features(D, F)
    for e in sorted(Fe, key=lambda e: (kind.get(e, ''), e)):
        f = Fe[e]
        print(f'{e} {kind.get(e, ""):<15} n={f["n"]:3d} tok={f["tok"]:6d} T={f["T"]:.3f} Thi={f["Thi"]:.3f} Tlo={f["Tlo"]:.3f} dT={f["Thi"]-f["Tlo"]:+.3f} top1={f["top1"]:.3f} tail={f["tail"]:.4f}'
              + (f' | fp8: T={f["Tf"]:.3f} LLR/ktok={f["llr"]:7.2f} z={f["z"]:6.2f}' if F else ''))
