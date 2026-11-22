"""Per-endpoint LLR(base - variant) per 1000 tokens with answer-level bootstrap SE, optionally only long answers.
Usage: wb_llr.py base_names variant_names [min_tokens]"""
import sys, numpy as np
from wb_fit import load, kind
from wb_feat import stack, Tfit, llT
rng = np.random.default_rng(0)
def stat(D, F, ks):
    Tb, Tf = Tfit(*stack(D, ks)), Tfit(*stack(F, ks))
    return 1000 * sum(llT(D[k], Tb).sum() - llT(F[k], Tf).sum() for k in ks) / sum(D[k]['n'] for k in ks)
if __name__ == '__main__':
    D, F = load(sys.argv[1]), load(sys.argv[2]); mn = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    out = {}
    for e in sorted({k[0] for k in D}):
        ks = [k for k in D if k[0] == e and k in F and D[k]['n'] >= mn]
        if len(ks) < 4: continue
        v = stat(D, F, ks); B = [stat(D, F, [ks[i] for i in rng.integers(0, len(ks), len(ks))]) for _ in range(100)]
        out[e] = (v, np.std(B)); print(f'{e} {kind.get(e, "")[:12]:<12} n={len(ks):3d} tok={sum(D[k]["n"] for k in ks):5d} llr={v:7.2f} ± {np.std(B):5.2f}', flush=True)
