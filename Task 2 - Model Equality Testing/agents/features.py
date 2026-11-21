"""Create anonymous cohorts from Plan R simulated long answers (no network)."""

import argparse
from collections import defaultdict
from itertools import combinations
import json
import math
import random
import re
import statistics as st
import tarfile
from pathlib import Path


def describe(text, prompt):
    words = re.findall(r"\w+", text.casefold())
    prefix = re.findall(r"\w+", prompt.split("\n")[-1].casefold())
    if prefix and words[: len(prefix)] == prefix:
        words = words[len(prefix) :]
    n = max(1, len(text))
    shapes = [
        sum(f(c) for c in text) / n
        for f in (
            str.isdigit,
            str.isupper,
            str.isspace,
            lambda c: c in ".,;!?",
            lambda c: c == "\n",
            lambda c: c in "#*`:-",
        )
    ]
    return set(words), words[:5], shapes, math.log1p(len(words))


def features(samples, eps):
    ids = set.intersection(*(set(samples[e]) for e in eps))
    if not ids:
        raise ValueError("No aligned long responses")
    values = []
    for e, f in combinations(eps, 2):
        per = []
        for p in sorted(ids):
            (w, x, a, n), (v, y, b, m) = samples[e][p], samples[f][p]
            per.append(
                (
                    1 - len(w & v) / max(1, len(w | v)),
                    sum(x[:i] != y[:i] for i in range(1, 6)) / 5,
                    math.dist(a, b),
                    abs(n - m),
                )
            )
        values.append(list(map(st.mean, zip(*per))))
    norms = [st.median(col) or 1 for col in zip(*values)]
    return [[round(v / d, 7) for v, d in zip(row, norms)] for row in values]


def cohort(rows, rng):
    rng.shuffle(rows)
    samples, labels = defaultdict(dict), {}
    for r in rows:
        e = r["endpoint"]
        samples[e][r["id"]] = describe(r["response"], r["prompt"])
        labels[e] = r["model"]
    eps = sorted(samples)
    rng.shuffle(eps)
    index = {}
    return {
        "labels": [index.setdefault(labels[e], len(index)) for e in eps],
        "features": features(samples, eps),
    }


def prepare(archive, names):
    data = {}
    with tarfile.open(archive, "r:gz") as t:
        for name in names:
            blocks = defaultdict(list)
            with t.extractfile(name + ".jsonl") as f:
                for row in map(json.loads, f):
                    if row.get("event") == "answer" and row.get("kind") == "long":
                        blocks[row["block"]].append(row)
            data[name] = [
                cohort(group, random.Random(f"{name}:{b}"))
                for b, group in sorted(blocks.items())
            ]
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("names", nargs="+")
    a = p.parse_args()
    a.output.write_text(
        json.dumps(prepare(a.archive, a.names), separators=(",", ":")) + "\n"
    )
