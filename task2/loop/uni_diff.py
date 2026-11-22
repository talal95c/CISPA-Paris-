"""Pairwise 'different weights' test from the unigram fingerprint: d_h = r_a,h/n_a,h - r_b,h/n_b,h on two halves of the answers,
z = <d_0, d_1> / sd (≈N(0,1) if a and b share weights; any shared misfit of the checkpoint cancels). Usage: uni_diff.py [--w raw|chi] uni/name ..."""
import sys, numpy as np
from wb_fit import kind
import uni_sim as U
def table(files, W='raw'):
    S, R, VV, T = U.load(files); eps = sorted(S); Z = {}
    n = {(e, h): S[e]['n'][e][h] for e in eps for h in (0, 1)}
    for i, a in enumerate(eps):
        for b in eps[i + 1:]:
            d = [R[a, h] / n[a, h] - R[b, h] / n[b, h] for h in (0, 1)]
            v = [VV[a, h] / n[a, h] ** 2 + VV[b, h] / n[b, h] ** 2 for h in (0, 1)]
            if W == 'raw': z = (d[0] * d[1]).sum() / np.sqrt((v[0] * v[1]).sum())
            else:
                k = 0.5 / (0.5 * (n[a, 0] + n[b, 0])) ** 2; s = np.sqrt((v[0] + k) * (v[1] + k))
                z = (d[0] * d[1] / s).sum() / np.sqrt((v[0] * v[1] / s ** 2).sum())
            Z[a, b] = Z[b, a] = z
    return eps, Z, T
if __name__ == '__main__':
    args = sys.argv[1:]; W = 'raw'
    if args[0] == '--w': W = args[1]; args = args[2:]
    eps, Z, T = table(args, W)
    print('       ' + ' '.join(e[-2:].rjust(5) for e in eps))
    for a in eps: print(f'  {a[-5:]} ' + ' '.join(('  .  ' if a == b else f'{Z[a, b]:5.1f}') for b in eps) + f'  {kind.get(a, "")[:14]} T={T[a]:.2f}')
