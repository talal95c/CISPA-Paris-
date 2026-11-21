"""Plan R structural decisions and exact joint expected MCC."""
import math
from collections import defaultdict
from itertools import combinations, product

from features import DEFAULTS, clipped_llr, pair_features


def mcc(tp, fp, fn, tn):
    denom = math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return max(0.0, (tp*tn-fp*fn) / denom) if denom else 0.0


def pairs(nodes):
    return frozenset(combinations(sorted(nodes), 2))


def confusion(truth, guess, total):
    tp = len(truth & guess)
    fp, fn = len(guess) - tp, len(truth) - tp
    return tp, fp, fn, total - tp - fp - fn


def score(truth, guess):
    universe = pairs(truth)
    family_true = {pair for pair in universe if truth[pair[0]][0] == truth[pair[1]][0]}
    family_guess = {pair for pair in universe if guess[pair[0]][0] == guess[pair[1]][0]}
    model_true = {pair for pair in family_true if truth[pair[0]] == truth[pair[1]]}
    model_guess = {pair for pair in family_true if guess[pair[0]] == guess[pair[1]]}
    family = mcc(*confusion(family_true, family_guess, len(universe)))
    model = mcc(*confusion(model_true, model_guess, len(family_true)))
    return {"family": family, "model": model, "total": .1*family + .9*model}


def pair_key(a, b):
    return tuple(sorted((a, b)))


def group_families(endpoints, llr, threshold=0.0):
    groups = [(e,) for e in sorted(endpoints)]
    while len(groups) > 1:
        ranked = [
            (sum(llr[pair_key(a, b)] for a in groups[i] for b in groups[j]) / (len(groups[i])*len(groups[j])), i, j)
            for i, j in combinations(range(len(groups)), 2)
        ]
        strongest, i, j = max(ranked, key=lambda t: (t[0], -t[1], -t[2]))
        if strongest < threshold:
            break
        merged = tuple(sorted((*groups[i], *groups[j])))
        groups = sorted([g for n, g in enumerate(groups) if n not in {i, j}] + [merged])
    return groups


def rule(anchors, candidates):
    return anchors | candidates if len(candidates) in (1, 2) else anchors


def family_proposals(group, llrs, strong, soft, candidate_floor=0.0):
    """Enumerate possible labels; no plain is privileged among the eligible candidates."""
    strongest_soft = sorted((ep for ep in group if ep in soft and ep not in strong),
                            key=lambda ep: (-soft[ep], ep))[:3]
    hard = set(group) & strong
    guesses, worlds = set(), defaultdict(float)
    for flags in product((False, True), repeat=len(strongest_soft)):
        anchors = hard | {e for e, yes in zip(strongest_soft, flags) if yes}
        probability = math.prod(soft[e] if yes else 1-soft[e] for e, yes in zip(strongest_soft, flags))
        candidates = {e for e in group if e not in anchors and anchors and
                      max(llrs[pair_key(e, a)] for a in anchors) >= candidate_floor}
        guesses.add(frozenset(rule(anchors, candidates)))
        if candidates:
            for plain in candidates:
                worlds[pairs(anchors | {plain})] += probability / len(candidates)
        else:
            worlds[pairs(anchors)] += probability
    return sorted(guesses, key=lambda g: (len(g), tuple(sorted(g)))), worlds


def optimize_joint(groups, proposals):
    """Combine local confusion-count distributions before taking MCC, not after."""
    local = []
    for group, (guesses, worlds) in zip(groups, proposals):
        total = len(pairs(group))
        choices = []
        for guess in guesses:
            guess_pairs = pairs(guess)
            outcomes = defaultdict(float)
            for truth_pairs, p in worlds.items():
                outcomes[confusion(truth_pairs, guess_pairs, total)] += p
            choices.append((guess, outcomes))
        local.append(choices)
    best = None
    for joint in product(*local):
        dist = {(0, 0, 0, 0): 1.0}
        for _, outcomes in joint:
            merged = defaultdict(float)
            for a, p in dist.items():
                for b, q in outcomes.items():
                    merged[tuple(x+y for x, y in zip(a, b))] += p*q
            dist = merged
        expected = sum(mcc(*counts)*p for counts, p in dist.items())
        if best is None or expected > best[0] + 1e-12:
            best = expected, [g for g, _ in joint]
    return best


def infer(sigs, params=None, settings=None):
    params = params or DEFAULTS
    settings = settings or {}
    dimensions = pair_features(sigs)
    burst_weight = settings.get("burst_weight", 0.35)
    families = {k: clipped_llr(v, params["family"], (0.75, 0.25, burst_weight)) for k, v in dimensions.items()}
    models = {k: clipped_llr(v, params["model"], (0.30, 0.15, burst_weight)) for k, v in dimensions.items()}
    groups = group_families(sigs, families)
    strong, soft = set(), {}
    for e, sig in sigs.items():
        strength = sig.flags["strength"]
        if strength >= 1.5 and settings.get("strong_flags", True):
            strong.add(e)
        elif strength >= 1.5:
            soft[e] = 0.70 if strength > 2 else 0.60
        temp_stat = settings.get("temp_stat", "disagreement")
        diff = sig.repeat[temp_stat]
        threshold = settings.get("temp_threshold")
        if (threshold is not None and diff is not None and diff >= threshold and
                (temp_stat == "shape" or sig.repeat["matched"] >= 4)):
            if settings.get("strong_temp", False):
                strong.add(e)
            else:
                soft[e] = max(soft.get(e, 0), 0.60)
    expected, choices = optimize_joint(groups, [
        family_proposals(group, models, strong, soft, settings.get("candidate_floor", 0.0)) for group in groups
    ])
    labels = {}
    for i, (group, shared) in enumerate(zip(groups, choices), 1):
        family = f"F{i}"
        for j, ep in enumerate(group, 1):
            labels[ep] = family, f"{family}-{'S' if ep in shared else j}"
    return labels, {"expected_model_mcc": expected, "families": groups, "strong": sorted(strong), "soft": soft}
