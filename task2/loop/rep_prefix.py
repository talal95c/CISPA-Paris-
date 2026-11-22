"""Watermark by repetition: mean number of identical leading words between two answers of the same endpoint to the same prompt.
Usage: rep_prefix.py file1,file2 [idprefix]"""
import sys, json, collections, itertools, numpy as np
rows = [json.loads(l) for f in sys.argv[1].split(',') for l in open(f)]
pre = sys.argv[2] if len(sys.argv) > 2 else ''
G = collections.defaultdict(dict)
for r in rows:
    if r.get('response') and r['id'].startswith(pre): G[r['endpoint'], r['id']][r['repeat']] = r['response'].split()
def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n
E = collections.defaultdict(list); X = collections.defaultdict(list); L = collections.defaultdict(list)
for (e, i), d in G.items():
    for a, b in itertools.combinations(sorted(d), 2): E[e].append(lcp(d[a], d[b])); X[e].append(d[a] == d[b])
    L[e] += [len(v) for v in d.values()]
for e in sorted(E): print(f'{e} pairs={len(E[e]):3d} lcp mean={np.mean(E[e]):5.1f} median={np.median(E[e]):4.0f} max={max(E[e]):3d} identical={np.mean(X[e]):.2f} len={np.mean(L[e]):5.0f}')
