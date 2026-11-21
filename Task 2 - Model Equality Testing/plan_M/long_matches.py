"""Conservative model grouping from matching continuations on shared prompts."""

from collections import defaultdict
from itertools import combinations

from journal import long_answers


MIN_EXACT_CHARS = 64
MIN_PREFIX_CHARS = 128


def prompt_matches(left, right, slots):
    """Return a witness for two independent matches, or the first match alone."""
    candidates = []
    for pid in sorted(left.keys() & right.keys()):
        a, b = (text.replace("\r\n", "\n").strip() for text in (left[pid], right[pid]))
        short, long = sorted((a, b), key=len)
        if a == b and len(short) >= MIN_EXACT_CHARS:
            kind = "exact"
        elif a != b and len(short) >= MIN_PREFIX_CHARS and long.startswith(short):
            kind = "prefix"
        else:
            continue
        candidates.append(((slots[pid].prompt, a, b, short), (pid, kind)))

    # Both prompts, both endpoints' replies and the shared text must differ.
    # A later repeat of a prompt may provide the independent match.
    for (keys_a, evidence_a), (keys_b, evidence_b) in combinations(candidates, 2):
        if all(a != b for a, b in zip(keys_a, keys_b)):
            return [evidence_a, evidence_b]
    return [candidates[0][1]] if candidates else []


def partition_long_matches(labels, rows, slots):
    answers, prompts = long_answers(rows, slots)
    families = defaultdict(list)
    for endpoint, (family, _) in sorted(labels.items()):
        families[family].append(endpoint)
    predictions, edges = {}, []
    for family, endpoints in sorted(families.items()):
        compatible = set()
        for left, right in combinations(endpoints, 2):
            matches = prompt_matches(answers.get(left, {}), answers.get(right, {}), prompts)
            if len(matches) == 2:
                compatible.add((left, right))
                edges.append({"endpoints": [left, right], "matches": matches})
        groups = []
        for endpoint in endpoints:
            # Require positive evidence for every pair. A chain of two links
            # does not establish that the two extremes are equal.
            group = next((group for group in groups if all((ep, endpoint) in compatible for ep in group)), None)
            if group is None:
                groups.append([endpoint])
            else:
                group.append(endpoint)
        for group in groups:
            model = f"{family}-long-{group[0]}"
            predictions.update({endpoint: (family, model) for endpoint in group})
    return predictions, edges
