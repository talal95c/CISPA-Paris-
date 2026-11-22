"""All white-box statistics per endpoint under its family's base checkpoint, split by prompt type.
Usage: wb_table.py base_names [variant_names ...]   (comma lists; each variant gives an LLR column: negative = variant fits better)"""
import sys, numpy as np
from wb_fit import load, GRID, kind
from wb_feat import stack, Tfit, llT
from wb_eos import caps, hazard
G = np.array(GRID)

def gof(D, ks, T):
    """goodness of fit at fitted T: top-1 excess z, dispersion of token log-lik, mean rank tail"""
    g = str(min(GRID, key=lambda t: abs(t - T)))
    r = np.concatenate([D[k]['rank'] for k in ks]); p1 = np.concatenate([D[k]['g'][g]['p1'] for k in ks])
    zx = np.concatenate([D[k]['zx'] for k in ks]); ez = np.concatenate([D[k]['g'][g]['ez'] for k in ks]); vz = np.concatenate([D[k]['g'][g]['vz'] for k in ks])
    return ((r == 0).sum() - p1.sum()) / np.sqrt((p1 * (1 - p1)).sum()), ((zx - ez) ** 2).sum() / vz.sum(), np.mean(r >= 100), len(r)

def llr(D, F, ks):
    kf = [k for k in ks if k in F]
    if len(kf) < 3: return float('nan'), float('nan'), 0
    Tb, Tf = Tfit(*stack(D, kf)), Tfit(*stack(F, kf))
    d = np.array([llT(D[k], Tb).sum() - llT(F[k], Tf).sum() for k in kf])
    return d.mean() / (d.std() + 1e-12) * np.sqrt(len(d)), 1000 * d.sum() / sum(D[k]['n'] for k in kf), len(kf)

if __name__ == '__main__':
    D = load(sys.argv[1]); Fs = [(n, load(n)) for n in sys.argv[2:]]; cap = caps(D)
    for e in sorted({k[0] for k in D}, key=lambda e: (kind.get(e, '') if e.startswith('ref') else '', e)):
        ks = [k for k in D if k[0] == e]; T = Tfit(*stack(D, ks)); z1, disp, tail, ntok = gof(D, ks, T)
        short = [k for k in ks if D[k]['n'] <= 12]; h = hazard(D, e, T, cap, set(ks))
        s = f'{e} {kind.get(e, "")[:14]:<14} n={len(ks):3d} tok={ntok:5d} T={T:.3f} z1={z1:5.1f} disp={disp:5.2f} tail={100*tail:4.1f}% stop_z={h["z"]:5.1f} early={h["early"]:5.1f}'
        for n, F in Fs:
            z, l, m = llr(D, F, ks); s += f' | {n.split(",")[0].split("_")[-1]}: z={z:5.2f} llr={l:6.2f} n={m}'
        print(s)
