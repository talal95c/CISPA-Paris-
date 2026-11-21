"""Conservative model grouping from matching continuations on shared prompts."""

from collections import defaultdict
from itertools import combinations

from journal import long_answers


MIN_MATCHED_PROMPTS = 2
MIN_EXACT_CHARS = 64
MIN_PREFIX_CHARS = 128


def prompt_matches(left, right, slots):
    evidence = {}
    seen_text = set()
    for pid in sorted(left.keys() & right.keys()):
        a, b = (text.replace("\r\n", "\n").strip() for text in (left[pid], right[pid]))
        short, long = sorted((a, b), key=len)
        prompt = slots[pid].prompt
        if short in seen_text or prompt in evidence:
            continue
        if a == b and len(short) >= MIN_EXACT_CHARS:
            evidence[prompt] = (pid, "exact")
        elif a != b and len(short) >= MIN_PREFIX_CHARS and long.startswith(short):
            evidence[prompt] = (pid, "prefix")
        else:
            continue
        seen_text.add(short)
    return list(evidence.values())


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
            if len(matches) >= MIN_MATCHED_PROMPTS:
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
