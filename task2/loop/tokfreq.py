"""Model-free watermark / weight test: token-frequency difference between two endpoints, split-half consistency.
d_h = f_a,h - f_b,h (token frequencies on half h of the (prompt, repeat) keys); z = sum_v d_0 d_1 / (fbar_v + eps), scaled by sqrt(n0 n1).
A watermark (fixed green list) or a weight change shifts token frequencies the same way in both halves. Usage: tokfreq.py tokenizer_dir files endpoints [minlen]"""
import sys, json, glob, collections, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1]); eps_ = sys.argv[3].split(','); minlen = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rows = [json.loads(l) for f in sys.argv[2].split(',') for l in open(f)]
rows = [r for r in rows if r['endpoint'] in eps_ and r.get('response')]
per = collections.defaultdict(set)
for r in rows: per[r['endpoint']].add((r['id'], r['repeat']))
common = set.intersection(*per.values()); rows = [r for r in rows if (r['id'], r['repeat']) in common]      # same prompts for everyone
keys = sorted(common); half = {k: i % 2 for i, k in enumerate(keys)}
C = {(e, h): collections.Counter() for e in eps_ for h in (0, 1)}
for r in rows:
    ids = tok(r['response'], add_special_tokens=False).input_ids[:256]
    if len(ids) >= minlen: C[r['endpoint'], half[r['id'], r['repeat']]].update(ids)
V = sorted({t for c in C.values() for t in c}); ix = {t: i for i, t in enumerate(V)}
F, N = {}, {}
for k, c in C.items():
    v = np.zeros(len(V)); 
    for t, n in c.items(): v[ix[t]] = n
    N[k] = v.sum(); F[k] = v / max(1, v.sum())
fbar = np.mean([F[k] for k in F], 0)
def z(a, b):
    d0, d1 = F[a, 0] - F[b, 0], F[a, 1] - F[b, 1]
    n0 = 1 / (1 / N[a, 0] + 1 / N[b, 0]); n1 = 1 / (1 / N[a, 1] + 1 / N[b, 1])
    w = 1 / (fbar + 2 / (N[a, 0] + N[b, 0]))
    return (d0 * d1 * w).sum() * np.sqrt(n0 * n1) / np.sqrt(len(V))
print('      ' + ' '.join(e[-2:].rjust(5) for e in eps_) + '   tokens')
for a in eps_: print(f'{a[-5:]} ' + ' '.join('  .  ' if a == b else f'{z(a, b):5.1f}' for b in eps_) + f'   {int(N[a, 0] + N[a, 1])}')
