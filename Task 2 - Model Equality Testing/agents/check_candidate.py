"""Offline evaluation of one frozen candidate; never recalibrate on audit sets."""

import argparse
import csv
from itertools import combinations
import json
from collections import defaultdict
from pathlib import Path

import arena
from features import describe, features

PROMPTS = {f"L{i:02}" for i in range(1, 10)}


def groups(journal, prior, expr, cutoff, repeat=0):
    by_family = defaultdict(list)
    samples = defaultdict(dict)
    with Path(prior).open() as f:
        for row in csv.DictReader(f):
            by_family[row["family"]].append(row["endpoint"])
    with Path(journal).open() as f:
        for row in map(json.loads, f):
            if row["repeat"] != repeat or row["id"] not in PROMPTS:
                continue
            cell = samples[row["endpoint"]]
            if row["id"] in cell:
                raise ValueError("Duplicate answer")
            cell[row["id"]] = describe(row["response"], row["prompt"])
    result = {}
    for family, eps in by_family.items():
        eps.sort()
        if any(set(samples[e]) != PROMPTS for e in eps):
            raise ValueError("Need L01-L09 for every endpoint in this repeat")
        clusters = [{i} for i in range(len(eps))]
        if len(eps) > 1:
            for h, partition in arena.merges(
                len(eps), features(samples, eps), arena.formula(expr)
            ):
                if h > cutoff:
                    break
                clusters = [s.copy() for s in partition]
        for i, cluster in enumerate(clusters):
            for j in cluster:
                result[eps[j]] = family, f"{family}_{i}"
    return result


def official(truth, prediction):
    if not truth or truth.keys() != prediction.keys():
        raise ValueError("Mismatched endpoint set")
    family, model = [0] * 4, [0] * 4
    for i, j in combinations(truth, 2):
        same = truth[i][0] == truth[j][0]
        pred = prediction[i][0] == prediction[j][0]
        family[2 * (not pred) + (not same)] += 1
        if same:
            model[2 * (prediction[i] != prediction[j]) + (truth[i] != truth[j])] += 1
    f, m = arena.mcc(*family), arena.mcc(*model)
    return dict(family=f, model=m, score=0.1 * f + 0.9 * m)


def audit(expr, cutoff, held, refs, key):
    if set(held) != {"holdout3", "holdout4"} or any(
        len(cs) != 5 for cs in held.values()
    ):
        raise ValueError("Expected five blocks each from holdout3 and holdout4")
    measure = arena.formula(expr)
    with Path(key).open() as f:
        truth = {r["endpoint"]: (r["family"], r["model"]) for r in csv.DictReader(f)}
    if set(truth) != {f"ref-{i:02}" for i in range(1, 10)}:
        raise ValueError("Expected reference key for ref-01..09")
    out = {
        name: arena.at([arena.states(c, measure) for c in cs], cutoff)
        for name, cs in held.items()
    }
    pred = groups(refs, key, expr, cutoff, repeat=1)
    out["reference_repeat_1"] = official(truth, pred)["model"]
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("finalist", "held", "refs", "reference_key"):
        p.add_argument(name, type=Path)
    p.add_argument(
        "--write-csv", type=Path, help="Create a new CSV only on a passing audit"
    )
    p.add_argument("--real-answers", type=Path)
    p.add_argument(
        "--previous-csv",
        type=Path,
        help="Use its estimated families, not its model labels",
    )
    a = p.parse_args()
    entry = arena.load(a.finalist)
    held = arena.load(a.held)
    base = audit(
        "lex",
        arena.evaluate("lex", arena.load())["cutoff"],
        held,
        a.refs,
        a.reference_key,
    )
    result = audit(entry["expr"], entry["cutoff"], held, a.refs, a.reference_key)
    eligible = all(x > max(0, base[k]) for k, x in result.items())
    print(json.dumps(dict(baseline=base, finalist=result, eligible=eligible), indent=2))
    if a.write_csv and eligible:
        if not (a.real_answers and a.previous_csv):
            p.error("--write-csv requires --real-answers and --previous-csv")
        pred = groups(a.real_answers, a.previous_csv, entry["expr"], entry["cutoff"])
        with a.previous_csv.open() as f:
            prior = {
                r["endpoint"]: (r["family"], r["model"]) for r in csv.DictReader(f)
            }
        if set(pred) != {f"ep-{i:02}" for i in range(1, 34)}:
            raise ValueError("Expected ep-01..ep-33")
        if any(
            (prior[e] == prior[f]) != (pred[e] == pred[f])
            for e, f in combinations(pred, 2)
        ):
            with a.write_csv.open("x", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["endpoint", "family", "model"])
                writer.writerows((e, *pred[e]) for e in sorted(pred))
        else:
            print("Same partition as the previous submission; no CSV written.")
