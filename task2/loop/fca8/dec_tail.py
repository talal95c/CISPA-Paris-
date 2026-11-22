"""Tail test at the pure-T MLE: tokens observed outside the top-p nucleus / top-k set versus the number expected under pure
temperature sampling from the base weights (expected ~ sum of -log Z over kept tokens)."""
import sys, numpy as np, math
from dec_an import at_T
for fn in sys.argv[1:]:
    z = np.load(fn, allow_pickle=True); TS, PS, KS = list(z['TS']), list(z['PS']), list(z['KS']); keys = z['keys'].tolist(); S = z['S']
    print('==', fn, '| observed/expected tokens outside the set, at the pure-T fit')
    for e in sorted({k[0] for k in keys}):
        E = S[[i for i, k in enumerate(keys) if k[0] == e]].sum(0)
        T, s0 = at_T(E[:, 0, 0], TS); i = int(np.searchsorted(TS, T)) - 1; i = max(0, min(i, len(TS) - 2)); w = (T - TS[i]) / (TS[i + 1] - TS[i])
        out = f"{e} T={T:.3f} tok={int(s0[0]):5d}"
        for pi, ki, name in [(1, 0, 'p.98'), (2, 0, 'p.95'), (3, 0, 'p.90'), (4, 0, 'p.80'), (0, 1, 'k100'), (0, 2, 'k50'), (0, 4, 'k20'), (0, 5, 'k10')]:
            s = E[i, pi, ki] * (1 - w) + E[i + 1, pi, ki] * w
            exp = (s[1] + s[12] - s0[1]) * s0[0] / max(s[0], 1)
            out += f" | {name} {s[2]:5.1f}/{exp:5.1f}"
        print(out)
