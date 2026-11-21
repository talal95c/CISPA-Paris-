"""CPU-only judge. Candidate input is an arithmetic expression, never executable Python."""

import ast
from itertools import combinations
import json
import math
import operator
from pathlib import Path
from statistics import mean

FIELDS = ("lex", "prefix", "shape", "length")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul}
FUN = {"min": min, "max": max, "abs": abs}


def formula(expr):
    if not isinstance(expr, str) or len(expr) > 400:
        raise ValueError("Expression must fit in 400 characters")
    tree = ast.parse(expr.strip(), mode="eval")
    if len(list(ast.walk(tree))) > 80:
        raise ValueError("Expression too complex")

    def build(node):
        if (
            isinstance(node, ast.Constant)
            and type(node.value) in (float, int)
            and abs(node.value) <= 10
        ):
            return lambda x: node.value
        if isinstance(node, ast.Name) and node.id in FIELDS:
            return lambda x: x[FIELDS.index(node.id)]
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            f, g = build(node.left), build(node.right)
            return lambda x: OPS[type(node.op)](f(x), g(x))
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FUN
        ):
            if node.keywords or len(node.args) not in (
                (1,) if node.func.id == "abs" else (2, 3, 4)
            ):
                raise ValueError("Invalid function arguments")
            args = [build(a) for a in node.args]
            return lambda x: FUN[node.func.id](*(a(x) for a in args))
        raise ValueError(
            "Only named distances, constants, +, -, *, abs, min, max allowed"
        )

    return build(tree.body)


def merges(n, features, measure):
    dist = {
        p: max(0.0, measure(v))
        for p, v in zip(combinations(range(n), 2), features, strict=True)
    }
    groups = [{i} for i in range(n)]
    yield -1.0, groups
    while len(groups) > 1:
        h, a, b = min(
            (mean(dist[tuple(sorted((i, j)))] for i in A for j in B), a, b)
            for (a, A), (b, B) in combinations(enumerate(groups), 2)
        )
        groups[a].update(groups.pop(b))
        yield h, groups


def states(cohort, measure):
    labels = cohort["labels"]
    pairs = list(combinations(range(len(labels)), 2))
    pos = sum(labels[i] == labels[j] for i, j in pairs)
    levels = []
    for h, groups in merges(len(labels), cohort["features"], measure):
        same = [(i, j) for G in groups for i, j in combinations(G, 2)]
        tp = sum(labels[i] == labels[j] for i, j in same)
        fp = len(same) - tp
        if levels and h == levels[-1][0]:
            levels.pop()
        levels.append((h, (tp, fp, pos - tp, len(pairs) - pos - fp)))
    return levels


def at(cuts, height):
    counts = [next(c for h, c in reversed(s) if h <= height) for s in cuts]
    tp, fp, fn, tn = map(sum, zip(*counts))
    return mcc(tp, fp, fn, tn)


def mcc(tp, fp, fn, tn):
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return (tp * tn - fp * fn) / math.sqrt(den) if den else 0.0


def fit(cuts):
    return max(sorted({h for s in cuts for h, _ in s}), key=lambda h: at(cuts, h))


def evaluate(expr, data):
    measure = formula(expr)
    groups = [[states(c, measure) for c in cohorts] for cohorts in data.values()]
    if len(groups) < 2 or any(not g for g in groups):
        raise ValueError("Need at least two development families")
    folds = [
        at(g, fit([s for k, other in enumerate(groups) if k != j for s in other]))
        for j, g in enumerate(groups)
    ]
    return {
        "mcc_by_family": folds,
        "objective": min(folds),
        "cutoff": fit([s for g in groups for s in g]),
    }


def load(path=None):
    return json.loads(
        Path(path or Path(__file__).with_name("development.json")).read_text()
    )
