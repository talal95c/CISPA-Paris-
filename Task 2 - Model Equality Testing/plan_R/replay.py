"""Evaluate Plan R on disjoint exact-budget simulation blocks."""
import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import replace
from itertools import combinations
from pathlib import Path

import numpy as np

from features import fit_gaussians, pair_features, read_rows, signatures
from partition import infer, score
from phase import temperature_result
from protocol import TYPES
from simulate import VARIANTS, replay_slots
from submit import write_csv


REMAINING_ITEMS = [f"{i}:{a}/{b}" for i, (_, options) in enumerate(TYPES[:-1])
                   for a, b in combinations(sorted(options), 2)]
ITEM_HOLDOUT = set(sorted(REMAINING_ITEMS, key=lambda key: hashlib.sha256(key.encode()).digest())
                   [: (len(REMAINING_ITEMS)*3 + 5)//10])


def holdout(item):
    return item.type == len(TYPES)-1 or item.key in ITEM_HOLDOUT


def load_blocks(paths):
    by_block = defaultdict(list)
    for path in paths:
        for row in read_rows(path)[1:]:
            if row.get("event") == "answer" and isinstance(row["block"], int):
                by_block[row["block"]].append(row)
    return by_block


def truth_rows(data):
    return {r["endpoint"]: ((r["family"], r["model"]), r["variant"]) for r in data}


def check_budget(data, slots):
    counter = Counter((r["endpoint"], r["id"]) for r in data)
    expected = {s.id: s.prompt for s in slots}
    endpoints = {r["endpoint"] for r in data}
    if len(endpoints) < 2*len(VARIANTS):
        raise ValueError("complete replay must have both local families")
    if any(counter[endpoint, key] != 1 for endpoint in endpoints for key in expected):
        raise ValueError("expected exactly one response per slot and endpoint")
    if any(r["id"] not in expected or r["prompt"] != expected[r["id"]] for r in data):
        raise ValueError("replay diverges from its frozen collection manifest")


def train(training, phase, leave_out=None):
    samples = {"family": [], "model": []}
    flag_counts, flag_totals = Counter(), Counter()
    distances = {key: defaultdict(list) for key in ("disagreement", "shape")}
    for block, sigs, truth in training:
        sigs = {ep: sig for ep, sig in sigs.items() if truth[ep][1] != leave_out}
        for (a, b), values in pair_features(sigs).items():
            fa, ma = truth[a][0]
            fb, mb = truth[b][0]
            samples["family"].append((values, fa == fb))
            if fa == fb:
                samples["model"].append((values, ma == mb))
        for ep, sig in sigs.items():
            name = truth[ep][1]
            flag_counts[name] += sig.flags["strength"] >= 1.5
            flag_totals[name] += 1
            for stat, results in distances.items():
                diff = sig.repeat[stat]
                if diff is not None and (stat == "shape" or sig.repeat["matched"] >= 4):
                    results[name].append((block, diff))
    flag_rates = {variant: flag_counts[variant]/count for variant, count in sorted(flag_totals.items())}
    strong_flags = (all(flag_rates.get(n, 0) >= .90 for n in ("system", "persona")) and
                    all(flag_rates.get(n, 1) <= .05 for n in ("fp8", "lora03", "lora07", "lora15") if n != leave_out))
    b3_pass = any(phase.get(size, {}).get("B3", {}).get("verdict") == "go" for size in ("20", "12"))
    strong_temp = b3_pass and temperature_result(distances["disagreement"])["verdict"] == "go"
    temp_stat = "disagreement" if strong_temp else "shape"
    values = distances[temp_stat]
    baseline = [x for n in ("plain", "system", "persona", "watermark") for _, x in values[n]]
    together = [x for _, x in values["temp"]]
    threshold = None
    if together and baseline:
        if strong_temp:
            threshold = (float(np.median(together)) + float(np.median(baseline))) / 2
        elif float(np.median(together)) > float(np.median(baseline)):
            threshold = float(np.quantile(baseline, .9))
    settings = {"strong_flags": strong_flags, "strong_temp": strong_temp, "temp_stat": temp_stat,
                "temp_threshold": threshold, "burst_weight": 0.65 if b3_pass else 0.15}
    return {task: fit_gaussians(examples, task) for task, examples in samples.items()}, settings, flag_rates


def summary(data):
    return {key: {"median": float(np.median([r[key] for r in data])),
                  "q10": float(np.quantile([r[key] for r in data], .10)),
                  "minimum": min(r[key] for r in data)}
            for key in ("family", "model", "total")}


def evaluate(blocks, size, phase):
    if len(blocks) < 5:
        raise ValueError("need at least five disjoint complete replay blocks")
    train_sigs, test_sigs, truth = {}, {}, {}
    for block, data in sorted(blocks.items()):
        slots = replay_slots(block, size)
        check_budget(data, slots)
        truth[block] = truth_rows(data)
        train_sigs[block] = signatures(data, slots, item_filter=lambda item: not holdout(item))
        test_sigs[block] = signatures(data, slots, item_filter=holdout)
        if set(test_sigs[block]) != set(truth[block]):
            raise ValueError(f"block {block} contains endpoint(s) with no usable text")
    baseline, noisy, ablations, predictions = [], [], defaultdict(list), {}
    exclusions = tuple(v for v in VARIANTS if v != "plain")
    for block, test in sorted(test_sigs.items()):
        training = [(b, sigs, truth[b]) for b, sigs in train_sigs.items() if b != block]
        params, settings, flag_rates = train(training, phase)
        labels, _ = infer(test, params, settings)
        predictions[block] = labels
        expected = {e: label for e, (label, _) in truth[block].items()}
        baseline.append(score(expected, labels))
        rng = random.Random(163 + block)
        disturbed = {}
        for ep, sig in sorted(test.items()):
            variant = truth[block][ep][1]
            triggered = rng.random() < flag_rates.get(variant, 0)
            flags = dict(sig.flags, strength=(max(1.8, sig.flags["strength"]) if triggered else 0.0))
            disturbed[ep] = replace(sig, flags=flags)
        noisy.append(score(expected, infer(disturbed, params, settings)[0]))
        for variant in exclusions:
            keep = {e: sig for e, sig in test.items() if truth[block][e][1] != variant}
            reduced = {e: expected[e] for e in keep}
            params_abl, settings_abl, _ = train(training, phase, leave_out=variant)
            ablations[variant].append(score(reduced, infer(keep, params_abl, settings_abl)[0]))
    aggregate = summary(baseline)
    good = aggregate["total"]["median"] >= .8 and aggregate["total"]["q10"] >= .6
    return {"n_blocks": len(baseline), "slots_per_endpoint": 100, "blocks": baseline,
            "summary": aggregate, "noisy_flags": summary(noisy), "noisy_flags_blocks": noisy,
            "leave_one_variant_out": {k: summary(v)["total"] for k, v in ablations.items()},
            "gate_C": "go" if good else "no-go"}, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journals", nargs="+", type=Path)
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--phase", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--practice-output", type=Path)
    args = parser.parse_args()
    phase = json.loads(args.phase.read_text()) if args.phase else {}
    result, predictions = evaluate(load_blocks(args.journals), args.items, phase)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.practice_output:
        first = predictions[min(predictions)]
        write_csv(args.practice_output, {e: label for e, label in first.items() if e.startswith("ref-")})
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
