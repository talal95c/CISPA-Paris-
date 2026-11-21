"""Network-free statistics, text ingestion and R/C model-equality decisions."""
import csv
import itertools
import json
import math
from pathlib import Path
import re

import numpy as np

VARIANTS = ['plain', 'fp8', 'lora0.3', 'lora0.7', 'lora1.5', 'system', 'persona', 'watermark', 'temp1']
SAME = {'plain', 'system', 'persona', 'watermark', 'temp1'}


def split_items(items):
    rng = np.random.default_rng(20261121)
    a, b, held = [], [], []
    for kind in range(6):
        indices = rng.permutation([i for i, x in enumerate(items) if x['kind'] == kind]).tolist()
        if kind == 5:
            held.extend(indices)
        else:
            a.extend(indices[:25])
            b.extend(indices[25:])
    return {name: rng.permutation(indices).tolist() for name, indices in [('tune', a), ('holdout', b), ('type', held)]}


def parse_burst(response, codebook):
    """Return one category per expected code; absent/duplicate/malformed = OTHER."""
    if not isinstance(response, str):
        raise ValueError('response must be text')
    result = {code: 10 for code in codebook}
    seen = set()
    for line in response.splitlines():
        match = re.fullmatch(r'\s*([a-z0-9]{3})\s+([^\r\n]+?)\s*', line)
        if not match or match[1] not in codebook:
            continue
        code, answer = match.groups()
        if code in seen:
            result[code] = 10
        else:
            options = codebook[code]['options']
            result[code] = options.index(answer) if answer in options else 10
        seen.add(code)
    return result


def read_answers(path, manifest):
    """Manifest maps prompt id to [{item, code, options, code_index}], half.

    Official answers.jsonl fields endpoint,id,repeat,response are accepted.
    Each prompt must list its expected items, including missing answers.
    Returns endpoint -> counts[half,item,code,category].
    """
    size = 1 + max(x['item'] for p in manifest.values() for x in p['items'])
    result, seen = {}, set()
    with Path(path).open() as handle:
        for line in handle:
            row = json.loads(line)
            key = (row['endpoint'], str(row['id']), row.get('repeat', 0))
            if key in seen:
                raise ValueError(f'Duplicate response key: {key}')
            seen.add(key)
            prompt = manifest[str(row['id'])]
            book = {x['code']: x for x in prompt['items']}
            answers = parse_burst(row['response'], book)
            counts = result.setdefault(row['endpoint'], np.zeros((2, size, 2, 11), dtype=np.int64))
            for code, category in answers.items():
                item = book[code]
                counts[prompt['half'], item['item'], item.get('code_index', 0), category] += 1
    return result


def u_items(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n, m = a.sum(-1), b.sum(-1)
    valid = (n >= 2) & (m >= 2)
    out = np.full(n.shape, np.nan)
    out[valid] = ((a[valid]*(a[valid]-1)).sum(-1)/(n[valid]*(n[valid]-1))
                  + (b[valid]*(b[valid]-1)).sum(-1)/(m[valid]*(m[valid]-1))
                  - 2*(a[valid]*b[valid]).sum(-1)/(n[valid]*m[valid]))
    return out


def distance(a, b):
    values = u_items(a, b)
    return float(np.nanmean(values)) if np.isfinite(values).any() else 0.


def standardized_u(a, b, null_mean, null_sd):
    if null_sd <= 0:
        raise ValueError('Null standard deviation must be positive')
    return (distance(a, b)-null_mean)/null_sd


def distances(counts):
    return np.array([[distance(a, b) if i != j else 0. for j, b in enumerate(counts)] for i, a in enumerate(counts)])


def frequencies(counts):
    return counts / np.maximum(counts.sum(-1, keepdims=True), 1)


def entropy_stat(counts):
    """Unbiased Gini dispersion; larger means less self-agreement."""
    n = counts.sum(-1)
    valid = n >= 2
    return float(np.mean(1-(counts[valid]*(counts[valid]-1)).sum(-1)/(n[valid]*(n[valid]-1)))) if valid.any() else 0.


def auc(negative, positive):
    delta = np.asarray(positive)[:, None]-np.asarray(negative)[None, :]
    return float(np.mean((delta > 0) + .5*(delta == 0)))


def mcc(truth, predicted):
    truth, predicted = np.asarray(truth, bool), np.asarray(predicted, bool)
    tp = int((truth & predicted).sum())
    fp = int((~truth & predicted).sum())
    fn = int((truth & ~predicted).sum())
    tn = int((~truth & ~predicted).sum())
    denominator = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return max(0., (tp*tn-fp*fn)/denominator) if denominator else 0.


def score(true_family, true_model, pred_family, pred_model):
    pairs = list(itertools.combinations(range(len(true_family)), 2))
    family = mcc([true_family[i] == true_family[j] for i, j in pairs], [pred_family[i] == pred_family[j] for i, j in pairs])
    intra = [(i, j) for i, j in pairs if true_family[i] == true_family[j]]
    model = mcc([true_model[i] == true_model[j] for i, j in intra], [(pred_family[i], pred_model[i]) == (pred_family[j], pred_model[j]) for i, j in intra])
    return dict(family=family, model=model, score=.1*family+.9*model)


def cluster(matrix, threshold):
    """Complete linkage: do not fuse families through a single noisy bridge."""
    groups = [[i] for i in range(len(matrix))]
    while len(groups) > 1:
        candidates = [(max(matrix[i, j] for i in a for j in b), x, y) for x, a in enumerate(groups) for y, b in enumerate(groups) if x < y]
        value, x, y = min(candidates)
        if value > threshold:
            break
        groups[x] += groups.pop(y)
    labels = np.zeros(len(matrix), dtype=int)
    for label, group in enumerate(groups):
        labels[group] = label
    return labels


def signed_projection(counts, pool, far):
    """Cross-fit: build direction on half h, score candidates on half 1-h."""
    out = np.zeros(len(counts))
    if not pool or not far:
        return out
    for h in [0, 1]:
        s = frequencies(counts[pool, h].sum(0))
        direction = frequencies(counts[far, h].sum(0))-s
        norm = np.linalg.norm(direction)
        if norm:
            direction /= norm
            other_pool = frequencies(counts[pool, 1-h].sum(0))
            out += ((frequencies(counts[:, 1-h])-other_pool)*direction).sum(axis=(1, 2))/2
    return out


def decide(counts, code_counts, noisy_flags, params, method='R'):
    """No true endpoint kinds/families enter this function.

    counts: endpoint,half,item,category (code-0 primary channel).
    code_counts: endpoint,item,code,category (probe channel).
    """
    total = counts.sum(1)
    matrix = (distances(total)-params['null_mean'])/params['null_sd']
    family = cluster(matrix, (params['family_threshold']-params['null_mean'])/params['null_sd'])
    labels = np.arange(len(counts), dtype=int)
    flags = np.asarray(noisy_flags, bool).copy()
    wm = np.array([distance(x[:, 0], x[:, 1]) for x in code_counts])
    temp = np.array([entropy_stat(x) for x in total])
    if params['enable_watermark']:
        flags |= wm > params['watermark_threshold']
    if params['enable_temperature']:
        for f in set(family):
            group = np.flatnonzero(family == f)
            flags[group] |= temp[group]-np.median(temp[group]) > params['temperature_threshold']
    next_label = len(labels)
    for f in set(family):
        group = np.flatnonzero(family == f).tolist()
        pool = [i for i in group if flags[i]]
        if not pool:
            continue
        labels[pool] = next_label
        pooled = total[pool].sum(0)
        candidates = [i for i in group if i not in pool and distance(total[i], pooled) <= params['pool_threshold']]
        far = [i for i in group if i not in pool and i not in candidates]
        k = len(candidates)
        selected = candidates if k in (1, 2) else []
        precision = params['rank_precision2'] if k == 2 else params['rank_precision3']
        if method == 'C' and k >= 2 and params['enable_ranking'] and precision > (.52 if k == 2 else .50):
            d = np.array([distance(total[i], pooled) for i in candidates])
            statistic = d
            if params['rank_stat'] == 'projection' and params['enable_axis']:
                statistic = signed_projection(counts, pool, far)[candidates]
            elif params['rank_stat'] == 'ranks' and params['enable_axis']:
                projection = signed_projection(counts, pool, far)[candidates]
                statistic = np.argsort(np.argsort(d)) + np.argsort(np.argsort(projection))
            selected = [candidates[int(np.argmin(statistic))]]
        labels[selected] = next_label
        next_label += 1
    return family, labels


def write_submission(path, endpoints, family, model):
    with Path(path).open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['endpoint', 'family', 'model'])
        writer.writerows(zip(endpoints, family, model))
