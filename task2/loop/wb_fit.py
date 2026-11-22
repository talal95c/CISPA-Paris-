"""Fit a sampling temperature per endpoint under each candidate checkpoint, then compare checkpoints by
log-likelihood ratio at their own fitted temperature. Usage: wb_fit.py nameA [nameB ...] [--ids prefix]"""
import json, sys, csv, numpy as np
GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4]
kind = {r['endpoint']: r['model'] + ':' + r['kind'][:12] for r in csv.DictReader(open('/home/ubuntu/met/practice_answers.csv'))}
try: kind.update({r['endpoint']: r['model'] for r in csv.DictReader(open('/home/ubuntu/ref-evaluation/audit/submission_audit.csv'))})
except Exception: pass

def load(n, keep=lambda k: True):
    D = {}
    for x in (json.loads(l) for m in n.split(',') for l in open(f'wb/{m}.jsonl')):
        k = (x['endpoint'], x['id'], x['repeat'])
        if keep(k): D[k] = x
    return D

def ll_by_T(x):
    zx = np.array(x['zx'])
    return np.array([(zx / T - np.array(x['g'][str(T)]['logZ'])).sum() for T in GRID])

def fit(D, e, keys=None):
    ks = [k for k in D if k[0] == e and (keys is None or k in keys)]
    L = np.array([ll_by_T(D[k]) for k in ks])            # answers x T
    tot = L.sum(0); i = int(tot.argmax())
    ntok = sum(D[k]['n'] for k in ks)
    T = GRID[i]; g = str(T)
    top1 = np.concatenate([np.array(D[k]['rank']) == 0 for k in ks])
    p1 = np.concatenate([D[k]['g'][g]['p1'] for k in ks])
    zx = np.concatenate([D[k]['zx'] for k in ks]); ez = np.concatenate([D[k]['g'][g]['ez'] for k in ks]); vz = np.concatenate([D[k]['g'][g]['vz'] for k in ks])
    z1 = (top1.sum() - p1.sum()) / np.sqrt((p1 * (1 - p1)).sum())
    disp = ((zx - ez) ** 2).sum() / vz.sum()
    return dict(keys=ks, L=L, i=i, T=T, ntok=ntok, ll=tot[i] / ntok, top1=top1.mean(), exp1=p1.mean(), z1=z1, disp=disp)

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    pref = next((a.split('=')[1] for a in sys.argv[1:] if a.startswith('--ids=')), None)
    keep = (lambda k: any(k[1].startswith(p) for p in pref.split(','))) if pref else (lambda k: True)
    Ds = [load(n, keep) for n in args]
    eps = sorted(set(k[0] for k in Ds[0]), key=lambda e: (kind.get(e, ''), e))
    for e in eps:
        F = [fit(D, e) for D in Ds]
        s = f'{e} {kind.get(e, ""):<15} tok={F[0]["ntok"]:6d}'
        for f in F: s += f' | T={f["T"]:.1f} ll={f["ll"]:6.3f} top1={f["top1"]:.3f}/{f["exp1"]:.3f} z1={f["z1"]:6.1f} disp={f["disp"]:.2f}'
        for f in F[1:]:
            common = [k for k in F[0]['keys'] if k in set(f['keys'])]
            ia = [F[0]['keys'].index(k) for k in common]; ib = [f['keys'].index(k) for k in common]
            d = F[0]['L'][ia, F[0]['i']] - f['L'][ib, f['i']]
            s += f' || LLR/ktok={1000 * d.sum() / F[0]["ntok"]:7.2f} z={d.mean() / (d.std() + 1e-12) * np.sqrt(len(d)):6.2f}'
        print(s)
