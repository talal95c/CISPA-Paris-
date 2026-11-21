"""Offline calibration/evaluation; tune reads ONLY 135M split A."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from decision import (VARIANTS, SAME, auc, decide, distance, distances,
                      entropy_stat, frequencies, score, signed_projection, split_items)

CACHE = Path.home() / 'me' / 'plan_c_cache'
ROOT = Path(__file__).resolve().parent


def load(family):
    paths = sorted(CACHE.glob(f'{family}-*.npz'))
    if len(paths) != 1:
        raise ValueError(f'Expected one base cache for {family}, got {paths}')
    with np.load(paths[0]) as data:
        meta = json.loads(str(data['metadata']))
        arrays = np.stack([data[k] for k in VARIANTS])
    return arrays, meta['items']


def schedule(indices, active, rng):
    selected = indices[:min(active, len(indices))]
    probes = selected[:22]
    n = len(selected)
    counts = np.zeros((2, n, 2), dtype=int)
    for half in range(2):
        # 23 primary bursts and eight paired-code bursts per half = 31 x 22.
        order = rng.permutation(n)
        for visit in range(23):
            counts[half, order[(np.arange(22)+visit*22) % n], 0] += 1
        counts[half, :22, :] += 4
    assert counts.sum() == 62*22
    return selected, probes, counts


def draw(probabilities, schedule_counts, rng):
    # probabilities: endpoint,item,code,category
    out = np.zeros((len(probabilities), 2, len(schedule_counts[0]), 2, 11), dtype=int)
    for h in range(2):
        for i in range(len(schedule_counts[h])):
            for c in range(2):
                n = int(schedule_counts[h, i, c])
                if n:
                    for e, p in enumerate(probabilities[:, i, c]):
                        out[e, h, i, c] = rng.multinomial(n, p/p.sum())
    return out


def probabilities(family, split, active, two_codes=True):
    base, items = load(family)
    indices = split_items(items)[split]
    indices = indices[:min(active, len(indices))]
    result = np.repeat(base[:, indices, None, :], 2, axis=2)
    if two_codes:
        with np.load(CACHE / f'codes-{family}-{split}.npz') as data:
            lookup = {int(x): i for i, x in enumerate(data['indices'])}
            for j, item in enumerate(indices[:22]):
                result[:, j, 1] = data['probabilities'][:, lookup[item]]
    return result, indices


def pilot():
    base, items = load('135m')
    indices = split_items(items)['tune']
    report = {}
    for active in [60, 125]:
        ids, _, sched = schedule(indices, active, np.random.default_rng(10))
        p = np.repeat(base[:, ids, None, :], 2, axis=2)
        rng = np.random.default_rng(711)
        null, alt, weak = [], [], []
        for _ in range(60):
            counts = draw(np.concatenate([p[:1], p[:1], p[1:2], p[2:3], p[4:5]]), sched, rng)[:, :, :, 0].sum(1)
            null.append(distance(counts[0], counts[1]))
            alt.append(distance(counts[0], counts[4]))
            weak.append(distance(counts[0], counts[2]))
        sd = float(np.std(null, ddof=1))
        report[active] = dict(null_mean=float(np.mean(null)), null_sd=sd, strong_z=float((np.mean(alt)-np.mean(null))/sd), fp8_z=float((np.mean(weak)-np.mean(null))/sd), strong_power=float(np.mean(np.asarray(alt)>np.quantile(null,.95))))
    print(json.dumps(report, indent=2))
    return report


def calibrate(active, repeats=200):
    p, ids = probabilities('135m', 'tune', active)
    _, _, sched = schedule(ids, active, np.random.default_rng(10))
    rng = np.random.default_rng(713)
    null, poolnull, familymax, wm, temp, ranks, cosines = [], [], [], [], [], {k: [] for k in ['distance', 'projection', 'ranks']}, []
    samples = []
    for r in range(repeats):
        # Independent duplicate plain for null; two independent plain endpoints form pool null.
        expanded = np.concatenate([p, p[:1], p[:1]])
        raw = draw(expanded, sched, rng)
        primary = raw[:, :, :, 0]
        total = primary.sum(1)
        null.append(distance(total[0], total[9]))
        poolnull.append(distance(total[0], total[9]+total[10]))
        familymax.append(float(distances(total[:9]).max()))
        wm.append([distance(x[:22, 0], x[:22, 1]) for x in raw[:9].sum(1)])
        dispersion = np.array([entropy_stat(x) for x in total[:9]])
        temp.append((dispersion-np.median(dispersion)).tolist())
        samples.append(primary[:9])
        # No endpoint truths used in decision(), but training labels define these diagnostics.
        pool = [5, 6]
        pooled = total[pool].sum(0)
        d = np.array([distance(x, pooled) for x in total[:9]])
        projection = signed_projection(primary[:9], pool, [4])
        for stat, values in [('distance', d), ('projection', projection)]:
            ranks[stat].append([int(np.argmin(values[c])) == 0 for c in [[0,1],[0,2],[0,1,2]]])
        combined = np.argsort(np.argsort(d)) + np.argsort(np.argsort(projection))
        ranks['ranks'].append([int(np.argmin(combined[c])) == 0 for c in [[0,1],[0,2],[0,1,2]]])
        if r < 20:
            freq = frequencies(total[:9])
            axis = freq[4]-freq[0]
            def cosine(x):
                delta = freq[x]-freq[0]
                den = np.linalg.norm(delta)*np.linalg.norm(axis)
                return float((axis*delta).sum()/den) if den else 0.
            cosines.append([cosine(x) for x in [2,3,5,6,7,8]])
    null, wm, temp = np.array(null), np.array(wm), np.array(temp)
    variants = [1,2,3,4]
    wm_threshold = float(np.quantile(wm[:, [i for i in range(9) if i != 7]], .99))
    wm_tpr = float(np.mean(wm[:,7]>wm_threshold))
    wm_fpr = np.mean(wm[:,variants]>wm_threshold, axis=0)
    t_threshold = float(np.quantile(temp[:, [i for i in range(9) if i != 8]], .99))
    t_auc = auc(temp[:,0], temp[:,8])
    t_fpr = np.mean(temp[:,variants]>t_threshold, axis=0)
    # Calibrate ranking using noisy observed flags, not an oracle S pool or LoRA label.
    ranks = {name: [] for name in ranks}
    flag_rng = np.random.default_rng(719)
    flag_draws = np.array([flag_rng.permutation(np.arange(repeats) < int((.9 if v in (5,6) else .05)*repeats)) for v in range(9)]).T
    wm_go = wm_tpr >= .9 and float(max(wm_fpr)) <= .05
    temp_go = t_auc >= .9 and float(max(t_fpr)) <= .10
    pool_threshold = float(np.quantile(poolnull,.95))
    for r, primary in enumerate(samples):
        observed = flag_draws[r].copy()
        if wm_go:
            observed |= wm[r] > wm_threshold
        if temp_go:
            observed |= temp[r] > t_threshold
        pool = np.flatnonzero(observed).tolist()
        total = primary.sum(1)
        if not pool:
            for name in ranks:
                ranks[name].append([False]*3)
            continue
        pooled = total[pool].sum(0)
        d = np.array([distance(x, pooled) for x in total])
        far = [i for i in range(9) if i not in pool and d[i] > pool_threshold]
        projection = signed_projection(primary, pool, far)
        # Endpoint-order-independent random tie breaks for the rank-sum diagnostic.
        combined = np.argsort(np.argsort(d)) + np.argsort(np.argsort(projection)) + flag_rng.uniform(0,.01,9)
        for name, values in [('distance',d),('projection',projection),('ranks',combined)]:
            ranks[name].append([not any(observed[c]) and int(np.argmin(values[c])) == 0 for c in [[0,1],[0,2],[0,1,2]]])
    rank_acc = {name: np.mean(value, axis=0).tolist() for name, value in ranks.items()}
    cosines = np.array(cosines)
    successes = int(np.sum((cosines[:,:2]>.8).all(1) & (cosines[:,2:]<.4).all(1)))
    axis_go = successes >= 18
    allowed = ['distance'] + (['projection','ranks'] if axis_go else [])
    chosen = max(allowed, key=lambda name: min(rank_acc[name]))
    rank_go = min(rank_acc[chosen]) >= .6
    base, items = load('135m')
    delta_items = int((np.max(np.abs(base[1,ids]-base[0,ids]), axis=1)>.05).sum())
    # Family threshold uses only observed within-family distances on A/135M.
    params = dict(schema=1, seed=20261121, tuning_family='135m', tuning_split='A without type 5', held_type=5,
                  active_items=active, probe_items=22, bursts=62, width=22, primary_bursts=46, code_bursts=16,
                  null_mean=float(null.mean()), null_sd=float(null.std(ddof=1)),
                  null_q95=float(np.quantile(null,.95)), pool_threshold=float(np.quantile(poolnull,.95)),
                  family_threshold=float(np.quantile(familymax,.95)), watermark_threshold=wm_threshold,
                  temperature_threshold=t_threshold, enable_watermark=wm_tpr>=.9 and float(max(wm_fpr))<=.05,
                  enable_temperature=t_auc>=.9 and float(max(t_fpr))<=.10, enable_axis=axis_go,
                  enable_ranking=rank_go, rank_stat=chosen, rank_precision2=min(rank_acc[chosen][:2]),
                  rank_precision3=rank_acc[chosen][2], noisy_flags_enabled={'0.7':False,'0.9':True},
                  calibration_replays=repeats)
    gates = dict(B2=dict(mean=params['null_mean'],sd=params['null_sd'],q95=params['null_q95'],go=bool(abs(params['null_mean'])<=3*params['null_sd']/np.sqrt(repeats))),
                 B4=dict(tpr=wm_tpr,fpr_variants=wm_fpr.tolist(),go=params['enable_watermark']),
                 flags={'q0.7':dict(tpr=.7,fpr=.05,go=False),'q0.9':dict(tpr=.9,fpr=.05,go=True)},
                 B6=dict(successes=successes,out_of=20,go=axis_go),
                 B7=dict(auc=t_auc,fpr_variants=t_fpr.tolist(),go=params['enable_temperature']),
                 R1=dict(precision=rank_acc,chosen=chosen,go=rank_go),
                 D4=dict(items=delta_items,go=delta_items>=5))
    return params, gates


def simulate_scores(families, split, params, q, repeats=200):
    active = params['active_items']
    data = [probabilities(f, split, active, two_codes=params['enable_watermark']) for f in families]
    rng = np.random.default_rng(821 + (0 if split=='tune' else 100))
    # Exact simulated flag rates across 200 replays per endpoint kind, not threshold fitting.
    noisy = np.empty((len(families), 9, repeats), bool)
    for f in range(len(families)):
        for v in range(9):
            positive = int(round((q if v in (5,6) else .05)*repeats))
            noisy[f,v] = rng.permutation(np.arange(repeats)<positive)
    results = {method: [] for method in ['R','C']}
    family_scores = []
    model_scores = {method:[] for method in results}
    for r in range(repeats):
        all_primary, all_codes, flags, true_family, true_model = [], [], [], [], []
        for f, (p, ids) in enumerate(data):
            _, _, sched = schedule(ids, active, rng)
            raw = draw(p, sched, rng)
            # 7–9 of nine variants; plain is always included. Same compositions for R/C/q.
            keep = [0]+sorted(rng.choice(np.arange(1,9), int(rng.integers(6,9)), replace=False).tolist())
            all_primary.extend(raw[keep,:,: ,0])
            all_codes.extend(raw[keep].sum(1)[:,:22])
            flags.extend(noisy[f,keep,r] if params['noisy_flags_enabled'][str(q)] else np.zeros(len(keep),bool))
            true_family.extend([f]*len(keep))
            true_model.extend([(f, 'S' if VARIANTS[v] in SAME else VARIANTS[v]) for v in keep])
        for method in results:
            family, model = decide(np.array(all_primary), np.array(all_codes), flags, params, method)
            result = score(true_family,true_model,family.tolist(),model.tolist())
            results[method].append(result['score'])
            model_scores[method].append(result['model'])
            if method == 'R':
                family_scores.append(result['family'])
    return {name: dict(median=float(np.median(values)),q10=float(np.quantile(values,.1)),model_median=float(np.median(model_scores[name]))) for name,values in results.items()} | {'family_median':float(np.median(family_scores))}


def audit_holdout(family, split, params, repeats=200):
    """Descriptive transfer checks only: NEVER modify frozen gates or thresholds."""
    p, ids = probabilities(family, split, params['active_items'], two_codes=params['enable_watermark'])
    _, _, sched = schedule(ids, params['active_items'], np.random.default_rng(10))
    rng=np.random.default_rng(939)
    null,temp,rank,cosines=[],[],[],[]
    for r in range(repeats):
        raw=draw(np.concatenate([p,p[:1]]),sched,rng)
        primary=raw[:9,:,:,0]
        total=raw[:,:,:,0].sum(1)
        null.append(distance(total[0],total[9]))
        dispersion=np.array([entropy_stat(x) for x in total[:9]])
        centered=dispersion-np.median(dispersion)
        temp.append(centered)
        observed=rng.random(9)<np.array([.9 if v in (5,6) else .05 for v in range(9)])
        if params['enable_temperature']:
            observed |= centered>params['temperature_threshold']
        if params['enable_watermark']:
            observed |= np.array([distance(x[:22,0],x[:22,1]) for x in raw[:9].sum(1)])>params['watermark_threshold']
        pool=np.flatnonzero(observed).tolist()
        if pool:
            d=np.array([distance(x,total[pool].sum(0)) for x in total[:9]])
            far=[i for i in range(9) if not observed[i] and d[i]>params['pool_threshold']]
            proj=signed_projection(primary,pool,far)
            statistic=d if params['rank_stat']=='distance' else proj
            if params['rank_stat']=='ranks':
                statistic=np.argsort(np.argsort(d))+np.argsort(np.argsort(proj))+rng.uniform(0,.01,9)
            rank.append([not any(observed[c]) and int(np.argmin(statistic[c]))==0 for c in [[0,1],[0,2],[0,1,2]]])
        else:
            rank.append([False]*3)
        if r<20:
            freq=frequencies(total[:9]);axis=freq[4]-freq[0]
            cos=[]
            for v in [2,3,5,6,7,8]:
                delta=freq[v]-freq[0];den=np.linalg.norm(axis)*np.linalg.norm(delta)
                cos.append(float((axis*delta).sum()/den) if den else 0.)
            cosines.append(cos)
    temp=np.array(temp);cosines=np.array(cosines)
    b,_=load(family)
    return dict(B2_mean=float(np.mean(null)),B2_sd=float(np.std(null,ddof=1)),
                B7_auc=auc(temp[:,0],temp[:,8]),B7_fpr_variants=np.mean(temp[:,[1,2,3,4]]>params['temperature_threshold'],axis=0).tolist(),
                R1_precision=np.mean(rank,axis=0).tolist(),B6_successes=int(((cosines[:,:2]>.8).all(1)&(cosines[:,2:]<.4).all(1)).sum()),
                D4_items=int((np.max(np.abs(b[1,ids]-b[0,ids]),axis=1)>.05).sum()),
                watermark='disabled by A/135M gate; no holdout model forwards spent' if not params['enable_watermark'] else 'enabled')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['pilot','freeze','evaluate'])
    args = parser.parse_args()
    if args.mode == 'pilot':
        pilot()
    elif args.mode == 'freeze':
        if (ROOT/'frozen_params.json').exists():
            raise RuntimeError('Refusing to overwrite frozen parameters')
        pilot_result = pilot()
        active = max(pilot_result,key=lambda k:pilot_result[k]['strong_z'])
        params,gates = calibrate(active)
        params['pilot'] = pilot_result
        frozen_text=json.dumps(params,indent=2)+'\n'
        gates_text=json.dumps(gates,indent=2)+'\n'
        (ROOT/'tuning_gates.json').write_text(gates_text)
        (ROOT/'frozen_params.json').write_text(frozen_text)
        print(json.dumps(dict(params=params,gates=gates),indent=2))
    else:
        # A committed freeze is a hard prerequisite for opening holdout distributions.
        import subprocess
        subprocess.run(['git','ls-files','--error-unmatch','frozen_params.json'],cwd=ROOT,check=True,capture_output=True)
        current = (ROOT/'frozen_params.json').read_bytes()
        committed = subprocess.check_output(['git','show','HEAD:Task 2 - Model Equality Testing/plan_C/frozen_params.json'],cwd=ROOT)
        if current != committed:
            raise RuntimeError('Frozen parameters differ from committed version')
        params=json.loads(current)
        out = dict(frozen_sha256=hashlib.sha256(current).hexdigest(),scores={})
        for split,families in [('tune',['135m']),('holdout',['360m']),('holdout',['qwen']),('holdout',['360m','qwen']),('type',['360m','qwen'])]:
            name=split+':'+','.join(families)
            out['scores'][name]={}
            for q in [.7,.9]:
                out['scores'][name][q]=simulate_scores(families,split,params,q)
                print(name,q,out['scores'][name][q],flush=True)
        # Three-family competition is a mixed diagnostic, NOT the unbiased holdout score.
        # It uses B (unseen during tuning) even for 135M.
        out['scores']['mixed-B:all']={q:simulate_scores(['135m','360m','qwen'],'holdout',params,q) for q in [.7,.9]}
        out['holdout_audits']={f:audit_holdout(f,'holdout',params) for f in ['360m','qwen']}
        (ROOT/'evaluation.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__ == '__main__':
    main()
