"""Calibrated base-vs-fp8 z by endpoint, split short prompts (probes_short.json ids) / other (long) / all."""
import json, sys, collections, math
SH = {p['id'] for p in json.load(open('probes_short.json'))}
for name in sys.argv[1:]:
    S = collections.defaultdict(lambda: collections.defaultdict(float))
    for l in open(f'wb/{name}.jsonl'):
        x = json.loads(l)
        short = x['id'] in SH or x['n'] <= 48
        for part in ('all', 'short' if short else 'long'):
            s = S[(x['endpoint'], part)]
            for k in ('n', 'obs', 'mu_b_1', 'v_b_1', 'mu_f_1', 'v_f_1'): s[k] += x[k]
    print('==', name, '| obs / attendu-base (par 1000 tok), z vs base, z vs fp8 simulee | court | long')
    for e in sorted({k[0] for k in S}):
        out = e
        for part in ('all', 'short', 'long'):
            s = S[(e, part)]
            if not s['n']: out += ' | -'; continue
            n = s['n'] / 1000
            out += f" | {part} tok={int(s['n']):5d} obs={s['obs']/n:6.1f}/{s['mu_b_1']/n:5.1f} zb={(s['obs']-s['mu_b_1'])/math.sqrt(s['v_b_1']):+5.1f} zf={(s['obs']-s['mu_f_1'])/math.sqrt(s['v_f_1']):+5.1f}"
        print(out)
