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

from features import read_rows, signatures
from partition import infer, score
from protocol import TYPES
from simulate import VARIANTS, replay_slots
from submit import FROZEN_CONFIG, write_csv


REMAINING_ITEMS = [f"{i}:{a}/{b}" for i, (_, options) in enumerate(TYPES[:-1])
                   for a, b in combinations(sorted(options), 2)]
ITEM_HOLDOUT = set(sorted(REMAINING_ITEMS, key=lambda key: hashlib.sha256(key.encode()).digest())
                   [: (len(REMAINING_ITEMS)*3 + 5)//10])


def holdout(item):
    return item.type == len(TYPES)-1 or item.key in ITEM_HOLDOUT


def load_blocks(paths, heldout=False):
    by_block = defaultdict(list)
    for path in paths:
        rows = read_rows(path)
        if not rows or bool(rows[0].get("heldout")) != heldout:
            raise ValueError(f"{path}: wrong prompt split")
        for row in rows[1:]:
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


def summary(data):
    return {key: {"median": float(np.median([r[key] for r in data])),
                  "q10": float(np.quantile([r[key] for r in data], .10)),
                  "minimum": min(r[key] for r in data)}
            for key in ("family", "model", "total")}


def prepare(blocks, config, heldout):
    if len(blocks) != config["blocks"]:
        raise ValueError("wrong number of disjoint complete replay blocks")
    sigs, truth, prompts = {}, {}, set()
    families = set(config["heldout_families" if heldout else "development_families"])
    for block, data in sorted(blocks.items()):
        slots = replay_slots(block, config["items"], heldout)
        check_budget(data, slots)
        if set(r["family"] for r in data) != families:
            raise ValueError(f"block {block} contains families outside the frozen split")
        prompts.update(s.prompt for s in slots)
        truth[block] = truth_rows(data)
        sigs[block] = signatures(data, slots, item_filter=lambda item: holdout(item) == heldout)
        if set(sigs[block]) != set(truth[block]):
            raise ValueError(f"block {block} contains endpoint(s) with no usable text")
    return sigs, truth, prompts


def evaluate(prepared, config, profile, flag_rates=None):
    signatures_by_block, truth, _ = prepared
    baseline, noisy, ablations, folds, predictions = [], [], defaultdict(list), defaultdict(list), {}
    params, settings = config["params"], config["profiles"][profile]
    for block, test in sorted(signatures_by_block.items()):
        labels, _ = infer(test, params, settings)
        predictions[block] = labels
        expected = {e: label for e, (label, _) in truth[block].items()}
        baseline.append(score(expected, labels))
        if flag_rates is not None:
            rng = random.Random(163 + block)
            disturbed = {}
            for ep, sig in sorted(test.items()):
                variant = truth[block][ep][1]
                triggered = rng.random() < flag_rates.get(variant, 0)
                flags = dict(sig.flags, strength=(max(1.8, sig.flags["strength"]) if triggered else 0.0))
                disturbed[ep] = replace(sig, flags=flags)
            noisy.append(score(expected, infer(disturbed, params, settings)[0]))
            for variant in (v for v in VARIANTS if v != "plain"):
                keep = {e: sig for e, sig in test.items() if truth[block][e][1] != variant}
                ablations[variant].append(score({e: expected[e] for e in keep}, infer(keep, params, settings)[0]))
            for family in config["heldout_families"]:
                keep = {e: sig for e, sig in test.items() if expected[e][0] == family}
                folds[family].append(score({e: expected[e] for e in keep}, infer(keep, params, settings)[0]))
    aggregate = summary(baseline)
    good = aggregate["total"]["median"] >= .8 and aggregate["total"]["q10"] >= .6
    return {"n_blocks": len(baseline), "slots_per_endpoint": 100, "blocks": baseline,
            "summary": aggregate, "noisy_flags": summary(noisy) if noisy else None,
            "leave_one_family_out_model": {k: summary(v)["model"] for k, v in folds.items()},
            "leave_one_variant_out": {k: summary(v)["total"] for k, v in ablations.items()},
            "gate_C": "go" if good else "no-go"}, predictions


def measure_flag_rates(prepared):
    sigs, truth, _ = prepared
    hits, totals = Counter(), Counter()
    for block, signatures_by_endpoint in sigs.items():
        for endpoint, signature in signatures_by_endpoint.items():
            variant = truth[block][endpoint][1]
            totals[variant] += 1
            hits[variant] += signature.flags["strength"] >= 1.5
    return {k: hits[k]/n for k, n in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", nargs="+", type=Path, required=True)
    parser.add_argument("--holdout", nargs="+", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=FROZEN_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--practice-output", type=Path)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config = json.loads(raw)
    if set(config["development_families"]) & set(config["heldout_families"]):
        parser.error("development and holdout families must be disjoint")
    if set(args.development) & set(args.holdout):
        parser.error("journal splits must be disjoint")
    dev = prepare(load_blocks(args.development), config, False)
    held = prepare(load_blocks(args.holdout, heldout=True), config, True)
    if dev[2] & held[2]:
        parser.error("exact prompts overlap between development and holdout")
    rates = measure_flag_rates(dev)
    result = {"config_sha256": hashlib.sha256(raw).hexdigest(), "profiles": {}}
    for profile in ("R", "simple"):
        development, predictions = evaluate(dev, config, profile)
        holdout_results, _ = evaluate(held, config, profile, rates)
        gap = development["summary"]["total"]["median"] - holdout_results["summary"]["total"]["median"]
        result["profiles"][profile] = {"development": development, "holdout": holdout_results, "gap": gap}
        if profile == "R" and args.practice_output:
            first = predictions[min(predictions)]
            write_csv(args.practice_output, {e: label for e, label in first.items() if e.startswith("ref-")})
    result["recommendation"] = ("simple" if abs(result["profiles"]["R"]["gap"]) >= config["gap_threshold"] else "R")
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
