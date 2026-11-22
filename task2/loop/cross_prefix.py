"""Cross-endpoint identical leading words on the same prompt (diagonal = between repeats of the same endpoint).
Usage: cross_prefix.py files endpoints [idprefix] [cap]"""
import sys, json, collections, itertools, numpy as np
rows = [json.loads(l) for f in sys.argv[1].split(',') for l in open(f)]
E = sys.argv[2].split(','); pre = sys.argv[3] if len(sys.argv) > 3 else ''; cap = int(sys.argv[4]) if len(sys.argv) > 4 else 30
G = collections.defaultdict(lambda: collections.defaultdict(list))
for r in rows:
    if r.get('response') and r['id'].startswith(pre) and r['endpoint'] in E: G[r['id']][r['endpoint']].append(r['response'].split())
def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return min(n, cap)
M = collections.defaultdict(list)
for i, d in G.items():
    for a in E:
        for b in E:
            if a == b: M[a, b] += [lcp(x, y) for x, y in itertools.combinations(d[a], 2)]
            else: M[a, b] += [lcp(x, y) for x in d[a] for y in d[b]]
print('      ' + ' '.join(e[-2:].rjust(5) for e in E))
for a in E: print(a[-5:], ' '.join(f'{np.mean(M[a, b]):5.1f}' if M[a, b] else '  -  ' for b in E))
