"""Text-only endpoint signatures and capped likelihood ratios."""
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations

import numpy as np

from protocol import manifest


PAIR_LINE = re.compile(r"(?<!\w)([A-Za-z]+)[ \t]+(?:\*\*)?([AB])(?:\*\*)?(?!\w)", re.I)
ROLE = re.compile(
    r"\b(?:pirate|captain|archivist|roleplay|character named|persona|in character|"
    r"virtual (?:librarian|teacher)|(?:speak|respond) as|arrgh?|matey)\b", re.I
)
INSTRUCTION = re.compile(
    r"\b(?:you are (?:a |an |the )|always (?:introduce|answer|respond|speak|follow)|"
    r"(?:your role|system (?:prompt|message)) (?:is|begins|says))\b", re.I
)
REFUSAL = re.compile(r"\b(?:can't|cannot|not able to|won't)\b", re.I)


def read_rows(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def parse_burst(text, items):
    tags = {item.tag for item in items}
    seen, duplicates = {}, set()
    for match in PAIR_LINE.finditer(text or ""):
        tag, pick = (x.lower() for x in match.groups())
        if tag not in tags:
            continue
        if tag in seen:
            duplicates.add(tag)
        seen[tag] = int(pick == "b")
    return {tag: pick for tag, pick in seen.items() if tag not in duplicates}


def canonical_choice(item, pick):
    return int(item.options[pick] == max(item.options))


def complete_format(text, items):
    lines = [line.strip() for line in (text or "").strip().splitlines() if line.strip()]
    return len(lines) == len(items) and all(re.fullmatch(rf"{item.tag} [AB]", line, re.I)
                                            for item, line in zip(items, lines))


def char_signature(text):
    text = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return Counter(text[i:i+3] for i in range(len(text) - 2))


def normalize(counts):
    norm = math.sqrt(sum(x*x for x in counts.values()))
    return {k: x/norm for k, x in counts.items()} if norm else {}


def cosine(a, b):
    return sum(x * b.get(key, 0) for key, x in a.items())


def flag_features(texts):
    identity = bool(ROLE.search(" ".join(texts.get(i, "") for i in ("flag-01", "flag-03"))))
    leak = texts.get("flag-02", "")
    leaked = len(INSTRUCTION.findall(leak)) >= 2 and not REFUSAL.search(leak)
    off_format = bool(texts.get("flag-04")) and texts["flag-04"].strip() != "FORMAT_OK"
    strength = float(identity) * 1.8 + float(leaked) * 1.5 + float(off_format and (identity or leaked)) * 0.5
    return {"identity": identity, "leak": leaked, "format": off_format, "strength": strength}


def dispersion(text_pairs, restrict_shape=False):
    differences, exact, shape = [], [], []
    for (first, second, items) in text_pairs:
        a, b = parse_burst(first, items), parse_burst(second, items)
        if restrict_shape:
            first = "\n".join(f"{item.tag} {'AB'[a[item.tag]]}" for item in items if item.tag in a)
            second = "\n".join(f"{item.tag} {'AB'[b[item.tag]]}" for item in items if item.tag in b)
        if first and second:
            exact.append(first.strip() == second.strip())
            shape.append(1 - cosine(normalize(char_signature(first)), normalize(char_signature(second))))
        differences.extend(a[tag] != b[tag] for tag in a.keys() & b.keys())
    return {"greedy": bool(len(exact) >= 2 and all(exact)),
            "disagreement": float(np.mean(differences)) if differences else None,
            "shape": float(np.mean(shape)) if shape else None, "matched": len(differences)}


def logit(value):
    return math.log(value / (1 - value))


def adjusted_odds(counts, greedy=False):
    if greedy:
        return ({key: 1.0 if hits * 2 >= n else -1.0 for key, (hits, n) in counts.items()}, 1.0)
    raw = {key: logit((hits + 0.5) / (n + 1)) for key, (hits, n) in counts.items()}
    reliable = [abs(raw[key]) for key, (_, n) in counts.items() if n >= 4]
    c = float(np.clip(np.sqrt(np.mean(np.square(reliable))), 0.5, 2.5)) if reliable else 1.0
    return ({key: value / c for key, value in raw.items()}, c)


@dataclass
class Signature:
    flags: dict
    long_by_prompt: dict
    long_style: dict
    odds: dict
    odds_count: dict
    c: float
    repeat: dict


def signatures(rows, slots=None, item_filter=None):
    slots = {s.id: s for s in (slots or manifest())}
    source = defaultdict(dict)
    for row in rows:
        if row.get("event") != "answer" or not (row.get("response") or "").strip():
            continue
        pid = row.get("probe_id", row["id"])
        if pid in slots:
            source[row["endpoint"]].setdefault(pid, row["response"])
    output = {}
    for endpoint, texts in source.items():
        long_counts = {pid: char_signature(text) for pid, text in texts.items() if slots[pid].kind == "long"}
        merged = Counter()
        for counts in long_counts.values():
            merged.update(counts)
        counts = defaultdict(lambda: [0, 0])
        for pid, text in texts.items():
            if slots[pid].kind != "burst":
                continue
            items = slots[pid].items
            response = parse_burst(text, items)
            for item in items:
                if item.tag in response and (item_filter is None or item_filter(item)):
                    counts[item.key][0] += canonical_choice(item, response[item.tag])
                    counts[item.key][1] += 1
        pairs = [
            (texts.get(f"burst-{i:02d}", ""), texts.get(f"burst-{i+60:02d}", ""),
             tuple(item for item in slots[f"burst-{i:02d}"].items if item_filter is None or item_filter(item)))
            for i in (1, 2)
        ]
        repeat = dispersion(pairs, restrict_shape=item_filter is not None)
        odds, c = adjusted_odds(counts, repeat["greedy"])
        output[endpoint] = Signature(
            flag_features(texts), {k: normalize(v) for k, v in long_counts.items()}, normalize(merged),
            odds, dict(counts), c, repeat,
        )
    return output


def pair_distance(a, b):
    prompts = a.long_by_prompt.keys() & b.long_by_prompt.keys()
    matched = [1 - cosine(a.long_by_prompt[p], b.long_by_prompt[p]) for p in sorted(prompts)]
    style = 1 - cosine(a.long_style, b.long_style) if a.long_style and b.long_style else None
    keys = sorted(a.odds.keys() & b.odds.keys())
    weights = [min(a.odds_count[k][1], b.odds_count[k][1]) for k in keys]
    # Greedy decompositions only supply their argmax, not a comparable probability.
    if a.repeat["greedy"] or b.repeat["greedy"]:
        squares = [float((a.odds[k] > 0) != (b.odds[k] > 0)) for k in keys]
    else:
        squares = [(a.odds[k] - b.odds[k]) ** 2 for k in keys]
    burst = float(np.sqrt(np.average(squares, weights=weights))) if sum(weights) >= 8 else None
    return (style, float(np.mean(matched)) if matched else None, burst)


# Each channel is a class-conditional Gaussian with common variance. Real reference
# calibration can replace the conservative startup estimates.
DEFAULTS = {
    "family": ((0.26, 0.60, 0.8), (0.55, 0.82, 1.4), (0.13, 0.15, 0.65)),
    "model": ((0.26, 0.60, 0.75), (0.36, 0.70, 1.05), (0.17, 0.19, 0.6)),
}


def clipped_llr(values, params, weights=(0.65, 0.35, 1.0)):
    a, b, sigma = params
    terms = [np.clip(((x-nu)**2 - (x-mu)**2) / (2*sd**2), -3, 3) * w
             for x, mu, nu, sd, w in zip(values, a, b, sigma, weights) if x is not None]
    return float(np.clip(sum(terms), -3, 3))


def pair_features(signatures_by_endpoint):
    return {(a, b): pair_distance(signatures_by_endpoint[a], signatures_by_endpoint[b])
            for a, b in combinations(sorted(signatures_by_endpoint), 2)}
