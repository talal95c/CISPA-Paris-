"""Equal-budget comparison without an LLM. Use only development data for selection."""

import argparse
from itertools import combinations
import json
import random

import arena


def control(budget, seed=42):
    pool = list(arena.FIELDS[1:])
    for x, y in combinations(arena.FIELDS, 2):
        pool.extend(
            [f"{x}+{y}", f"{x}*{y}", f"abs({x}-{y})", f"min({x},{y})", f"max({x},{y})"]
        )
    rng = random.Random(seed)
    data = arena.load()
    attempts = [
        dict(expr=s, **arena.evaluate(s, data))
        for s in ["lex", *rng.sample(pool, budget - 1)]
    ]
    return dict(best=max(attempts, key=lambda r: r["objective"]), attempts=attempts)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("budget", type=int, choices=range(1, 21))
    a = p.parse_args()
    print(json.dumps(control(a.budget), indent=2))
