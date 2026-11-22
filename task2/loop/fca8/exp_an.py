import json, sys, collections, math
for name in sys.argv[1:]:
    S = collections.defaultdict(lambda: collections.defaultdict(float))
    for l in open(f'wb/{name}.jsonl'):
        x = json.loads(l); s = S[x['endpoint']]
        for k, v in x.items():
            if isinstance(v, (int, float)) and k not in ('repeat', 'T'): s[k] += v
        s['rows'] += 1
    print('==', name, '| LLR par 1000 tokens : observe, attendu si base, attendu si fp8 simulee ; z = (obs - attendu)/ecart-type')
    for e in sorted(S):
        s = S[e]; n = s['n'] / 1000
        o = s['obs'] / n
        f = lambda h, p: ((s['obs'] - s[f'mu_{h}_{p}']) / math.sqrt(s[f'v_{h}_{p}']), s[f'mu_{h}_{p}'] / n)
        zb, mb = f('b', '1'); zf, mf = f('f', '1'); zbp, mbp = f('b', 'p90'); zfp, mfp = f('f', 'p90')
        print(f"{e} rows={int(s['rows']):3d} tok={int(s['n']):5d} obs={o:6.2f} | base {mb:6.2f} z={zb:+5.2f} | fp8 {mf:6.2f} z={zf:+5.2f} | top-p 0.9: base {mbp:6.2f} z={zbp:+5.2f} | fp8 {mfp:6.2f} z={zfp:+5.2f}")
