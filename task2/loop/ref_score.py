"""Score the white-box rule on the practice references with the official scorer.
Rule (thresholds fixed before scoring): stop-hazard z < ZL -> LoRA (own model); else fp8 LLR z < ZF -> fp8; else base.
Usage: ref_score.py [base=r360e] [fp8=r360_fp8]"""
import sys, subprocess, itertools, math, csv, collections, numpy as np
from wb_fit import load
from wb_feat import stack, Tfit, llT
from wb_eos import caps, hazard
ZL, ZF = -4.0, 0.0
OFF = '/home/ubuntu/met/score_practice.py'
truth = {r['endpoint']: r['model'] for r in csv.DictReader(open('/home/ubuntu/met/practice_answers.csv'))}
D = load(sys.argv[1] if len(sys.argv) > 1 else 'r360e'); F = load(sys.argv[2] if len(sys.argv) > 2 else 'r360_fp8')
cap = caps(D); eps = sorted(truth); ids = sorted(cap)

def feats(keys):
    out = {}
    for e in eps:
        ks = [k for k in keys if k[0] == e and k in F]
        T = Tfit(*stack(D, ks)); Tf = Tfit(*stack(F, ks))
        h = hazard(D, e, T, cap, set(ks))
        d = np.array([llT(D[k], T).sum() - llT(F[k], Tf).sum() for k in ks])
        out[e] = dict(n=len(ks), T=T, zs=h['z'], zf=d.mean() / (d.std() + 1e-12) * np.sqrt(len(d)))
    return out

def predict(f, zl=ZL, zf=ZF):
    return {e: ('L-' + e if v['zs'] < zl else 'Q-' + e if v['zf'] < zf else 'S') for e, v in f.items()}

def mcc(pred):
    tp = fp = fn = tn = 0
    for a, b in itertools.combinations(eps, 2):
        t = truth[a] == truth[b]; p = pred[a] == pred[b]
        tp += t and p; fp += p and not t; fn += t and not p; tn += (not t) and (not p)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return max(0.0, (tp * tn - fp * fn) / den) if den else 0.0

def official(pred, name):
    p = f'/tmp/{name}.csv'
    with open(p, 'w') as fh:
        fh.write('endpoint,family,model\n'); [fh.write(f'{e},P,{m}\n') for e, m in pred.items()]
    return subprocess.run([sys.executable, OFF, p], capture_output=True, text=True).stdout.strip().splitlines()

if __name__ == '__main__':
    sets = {'rep0': [k for k in D if k[2] == 0], 'rep1': [k for k in D if k[2] == 1],
            'promptsA': [k for k in D if ids.index(k[1]) % 2 == 0], 'promptsB': [k for k in D if ids.index(k[1]) % 2 == 1]}
    for name, ks in sets.items():
        f = feats(ks); p = predict(f)
        print(f'== {name}: mine={mcc(p):.3f} official: {official(p, "ref_" + name)[:1]}')
        for e in eps: print(f'   {e} {truth[e]:<3} n={f[e]["n"]:3d} T={f[e]["T"]:.3f} stop_z={f[e]["zs"]:6.1f} fp8_z={f[e]["zf"]:6.2f} -> {p[e]}')
    # endpoint-sized draws: one answer per prompt (36 answers per reference), thresholds fixed
    rng = np.random.default_rng(0); byp = collections.defaultdict(list)
    for k in D: byp[(k[0], k[1])].append(k)
    for npr in (36, 24, 12):
        m = []; err = collections.Counter()
        for _ in range(200):
            pi = set(rng.choice(ids, npr, replace=False))
            ks = [v[rng.integers(len(v))] for (e, i), v in byp.items() if i in pi]
            p = predict(feats(ks)); m.append(mcc(p))
            for e in eps: err[e] += (p[e] == 'S') != (truth[e] == 'S')
        m = np.array(m)
        print(f'one answer per prompt, {npr} prompts, 200 draws: MCC mean={m.mean():.3f} median={np.median(m):.3f} min={m.min():.3f} share=1.0: {np.mean(m > .999):.2f} | wrong class per ref: {dict(err)}')
