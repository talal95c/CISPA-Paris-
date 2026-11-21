"""Calculate a conditional MCC from FP8-vs-bf16 replay JSON files."""

import argparse
from itertools import combinations
import json
from math import prod, sqrt
from pathlib import Path
from statistics import median, quantiles


def mcc(pairs) -> float:
    counts = {(True, True): 0, (False, False): 0, (False, True): 0, (True, False): 0}
    for pair in pairs:
        counts[pair] += 1
    tp, tn, fp, fn = counts.values()
    denominator = sqrt(prod((tp + fp, tp + fn, tn + fp, tn + fn)))
    return max(0.0, (tp * tn - fp * fn) / denominator) if denominator else 0.0


def conditional_score(duels: list[tuple[int, int]]) -> float:
    # Plain, FP8, four disguised deployments, three LoRA. Family membership
    # and the seven labels outside the duel come from the simulation truth.
    truth = (0, 1, 0, 0, 0, 0, 2, 3, 4)
    rows = [
        (family, label, prediction)
        for family, duel in enumerate(duels)
        for label, prediction in zip(truth, duel + truth[2:], strict=True)
    ]
    pairs = list(combinations(rows, 2))
    family_mcc = mcc((a[0] == b[0], a[0] == b[0]) for a, b in pairs)
    model_mcc = mcc((a[1] == b[1], a[2] == b[2]) for a, b in pairs if a[0] == b[0])
    return 0.1 * family_mcc + 0.9 * model_mcc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="One gate JSON per synthetic family")
    paths = parser.parse_args().results
    runs = [json.loads(path.read_text()) for path in paths]
    if len(runs) < 2 or len({run["checkpoint"] for run in runs}) != len(runs):
        parser.error("Use at least two different synthetic families")
    if any(run["left_variant"] != "fp8" for run in runs):
        parser.error("Only duels with FP8 on the left and bf16 on the right can be scored")
    if len({len(run["margins_nats"]) for run in runs}) != 1 or len(runs[0]["margins_nats"]) < 2:
        parser.error("Require the same number of disjoint blocks (at least two) per family")
    scores = [
        conditional_score([
            (0, 1) if margin > 0 else (1, 0) if margin < 0 else (0, 0)
            for margin in block
        ])
        for block in zip(*(run["margins_nats"] for run in runs), strict=True)
    ]
    print(json.dumps({
        "median": median(scores),
        "q10": quantiles(scores, n=10, method="inclusive")[0],
        "minimum": min(scores),
        "scores": scores,
        "attach_both_baseline": conditional_score([(0, 0)] * len(runs)),
    }, indent=2))


if __name__ == "__main__":
    main()
