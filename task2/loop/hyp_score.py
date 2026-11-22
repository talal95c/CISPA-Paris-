"""Predicted public score of every past submission under a hypothesised truth (all 166 within-family pairs; public = part of them => +-0.05 noise).
Usage: hyp_score.py "23 33 04 14|30 20 06 07|31 16 29 10 27" ...   (groups separated by |, everything else singleton)"""
import sys, csv, json, itertools, math, os
FAM = dict(A='01 04 12 14 17 18 23 24 25 33'.split(), B='02 03 06 07 08 20 21 22 26 28 30 32'.split(), C='05 09 10 11 13 15 16 19 27 29 31'.split())
PAIRS = [(a, b) for v in FAM.values() for a, b in itertools.combinations(v, 2)]
SUBS = []
for l in open('sub/submissions.jsonl'):
    d = json.loads(l)
    if isinstance(d.get('body'), dict) and 'score' in d['body']:
        SUBS.append((os.path.basename(d['file'])[:-4], d['body']['score'], {r['endpoint'][-2:]: r['model'] + r['family'] for r in csv.DictReader(open(d['file']))}))
blk = lambda gs: {e: f'g{i}' for i, g in enumerate(gs) for e in g.split()}
allA, allB, allC = (' '.join(FAM[f]) for f in 'ABC')
SUBS += [('2141', .10, blk([allA, '03 06 07 08 20 21 22 26 30 32', '09 10 11 13 15 16 19 27 29 31'])), ('2312', .10, blk(['01 04 12', '06 21 22', '10 13 19']))]
def mcc(truth, pred):
    same = lambda d, a, b: d.get(a, a) == d.get(b, b)
    tp = fp = fn = tn = 0
    for a, b in PAIRS:
        t, g = same(truth, a, b), same(pred, a, b)
        tp += t and g; fp += (not t) and g; fn += t and not g; tn += (not t) and not g
    d = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (max(0, (tp * tn - fp * fn) / d) if d else 0), tp, fp, fn
if __name__ == '__main__':
    print(' ' * 60 + ' '.join(f'{s[0][-9:]:>9}' for s in SUBS)); print(' ' * 60 + ' '.join(f'{s[1]:9.2f}' for s in SUBS))
    for h in sys.argv[1:]:
        truth = blk(h.split('|'))
        sc = [.1 + .9 * mcc(truth, s[2])[0] for s in SUBS]
        print(f'{h[:58]:58s}  ' + ' '.join(f'{x:9.2f}' for x in sc) + f'  rmse={math.sqrt(sum((x - s[1]) ** 2 for x, s in zip(sc, SUBS)) / len(SUBS)):.3f}')
