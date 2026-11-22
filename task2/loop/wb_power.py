"""Which prompts carry the fp8 signal? Per prompt id: LLR(base - fp8sim) per answer for the fp8 reference vs the same-as-base ones. Usage: wb_power.py"""
import numpy as np, collections
from wb_fit import load, kind
from wb_feat import stack, Tfit, llT
D, F = load('r360e'), load('r360_fp8')
eps = sorted({k[0] for k in D})
Tb = {e: Tfit(*stack(D, [k for k in D if k[0] == e])) for e in eps}; Tf = {e: Tfit(*stack(F, [k for k in F if k[0] == e])) for e in eps}
d = {k: llT(D[k], Tb[k[0]]).sum() - llT(F[k], Tf[k[0]]).sum() for k in D if k in F}
by = collections.defaultdict(lambda: collections.defaultdict(list))
for k, v in d.items(): by[k[1].rstrip('0123456789')][k[0]].append((v, D[k]['n']))
for pid, g in sorted(by.items()):
    q = np.array([v for v, n in g.get('ref-02', [])]); o = np.array([v for e in g if kind[e].startswith('S') for v, n in g[e]])
    if len(q) < 2 or len(o) < 4: continue
    ntok = np.mean([n for e in g for v, n in g[e]])
    print('d=%5.2f %-10s nq=%3d no=%3d  fp8ref=%7.3f  S=%7.3f  tok=%5.1f  d/sqrt(tok)*16=%5.2f' % ((o.mean() - q.mean()) / np.sqrt((o.var() + q.var()) / 2 + 1e-9), pid, len(q), len(o), q.mean(), o.mean(), ntok, (o.mean() - q.mean()) / np.sqrt((o.var() + q.var()) / 2 + 1e-9) / np.sqrt(ntok) * 16))
