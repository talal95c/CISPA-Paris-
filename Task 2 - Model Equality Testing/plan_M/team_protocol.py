"""Adapter for the team's identity probes and long-completion journal."""

import re
from dataclasses import replace
from statistics import median

from journal import normalize_rows


FLAGS = {f"F{n}": f"flag-{n:02d}" for n in (1, 2, 3)}
FLAGS["F4"] = "team-primes"
LONGS = {f"L{n:02d}": f"long-{n:02d}" for n in range(1, 30)}
SLOTS = FLAGS | LONGS

FAMILY_MARKERS = {
    "Qwen": re.compile(r"\b(?:qwen(?=\d|\W|$)|alibaba(?: cloud)?\b)", re.I),
    "Llama": re.compile(r"\b(?:llama(?=\d|\W|$)|meta (?:ai|platforms)\b|(?<=by )meta\b)", re.I),
    "Anthropic": re.compile(r"\b(?:anthropic|claude)(?=\d|\W|$)", re.I),
}
FIRST_PERSON = re.compile(r"\b(?:i am|i['’]m|i was|my name|known as)\b", re.I)
CLAIM = re.compile(
    r"\b(?:i am|i['’]m|known as|my name is|created by|developed by|trained by|built by|designed by)"
    r"\s+(?:(?:an?|the)\s+)?\**$", re.I,
)
NEGATED = re.compile(r"\b(?:not|never|neither)\b[^,;.]*$", re.I)
MIN_SHARED_LONGS = 3
MIN_FAMILY_GAP = 0.03


def team_prompt_id(identifier):
    if isinstance(identifier, str) and re.fullmatch(r"L[0-9]{1,2}", identifier):
        return f"L{int(identifier[1:]):02d}"
    return identifier


def team_journal(source, prompt_manifest, protocol, *, allow_partial, expected):
    prompts = {}
    for row in prompt_manifest:
        pid = team_prompt_id(row["id"])
        if pid not in SLOTS or pid in prompts or not isinstance(row["prompt"], str) or not row["prompt"]:
            raise ValueError(f"Invalid team manifest entry: {pid!r}")
        prompts[pid] = row["prompt"]
    if prompts and prompts.keys() != SLOTS.keys():
        raise ValueError("The team manifest must list F1–F4 and L01–L29")

    entries = []
    for row in source:
        event = row.get("event")
        if event not in (None, "answer", "rejected"):
            continue
        pid = team_prompt_id(row.get("prompt_id", row.get("id")))
        if not isinstance(pid, str) or pid not in SLOTS:
            raise ValueError(f"Unknown team prompt ID: {pid!r}")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt or prompts.setdefault(pid, prompt) != prompt:
            raise ValueError(f"Team prompt {pid} does not match the manifest")
        if event != "rejected" and "response" not in row:
            raise ValueError(f"Team prompt {pid} is missing response")
        entries.append(dict(endpoint=row.get("endpoint_id", row.get("endpoint")),
                            id=SLOTS[pid], event=event or "answer",
                            prompt=prompt, response=row.get("response")))
    slots = [protocol.Slot(slot, "long" if pid in LONGS else "flag", 1, prompts.get(pid, ""))
             for pid, slot in SLOTS.items()]
    if not entries and allow_partial and expected is not None:
        return [], sorted(expected), slots
    rows, endpoints = normalize_rows(entries, slots, allow_partial=allow_partial)
    if expected is not None:
        if set(endpoints) - set(expected) or (not allow_partial and set(expected) - set(endpoints)):
            raise ValueError("The journal endpoint set does not match the target")
        endpoints = sorted(expected)
    return rows, endpoints, slots


def identity_family(response):
    """Use an unambiguous self-identification in the answer to F1."""
    disclosure = response[:450].strip()
    if not FIRST_PERSON.search(disclosure):
        return None
    mentions = {family: tuple(regex.finditer(disclosure)) for family, regex in FAMILY_MARKERS.items()}
    present = [family for family, matches in mentions.items() if matches]
    if len(present) == 1:
        family = present[0]
        for match in mentions[family]:
            before = disclosure[:match.start()]
            if CLAIM.search(before) and not NEGATED.search(before):
                return family
    return None


def signature_slots(slots):
    # R's feature extractor reads the two burst metadata records for its
    # repeat statistic. Empty item lists leave this channel unavailable.
    return [*slots, *(replace(slots[0], id=pid, kind="burst", prompt="")
                      for pid in ("burst-01", "burst-02"))]


def estimate_family(signature, grouped_anchors, features):
    distances = []
    for family, anchors in grouped_anchors.items():
        samples = []
        for anchor in anchors:
            if len(signature.long_by_prompt.keys() & anchor.long_by_prompt.keys()) < MIN_SHARED_LONGS:
                continue
            style, matched, _ = features.pair_distance(signature, anchor)
            if style is not None and matched is not None:
                samples.append(0.75 * style + 0.25 * matched)
        if len(samples) >= 2:
            distances.append((median(samples), family))
    distances.sort()
    if len(distances) >= 2 and distances[1][0] - distances[0][0] >= MIN_FAMILY_GAP:
        return distances[0][1]
    return None


def infer_team(rows, signatures, features, partition, params, settings):
    identities = {row["endpoint"]: identity_family(row["response"]) for row in rows
                  if row["id"] == "flag-01" and row["event"] == "answer"}
    groups = {name: {} for name in FAMILY_MARKERS}
    for endpoint, signature in signatures.items():
        claim = identities.get(endpoint)
        if claim:
            groups[claim][endpoint] = signature
    anchors = {family: tuple(group.values()) for family, group in groups.items() if group}
    unknown = {}
    for endpoint, signature in signatures.items():
        if identities.get(endpoint):
            continue
        estimate = estimate_family(signature, anchors, features)
        if estimate:
            groups[estimate][endpoint] = signature
        else:
            unknown[endpoint] = signature

    labels, families = {}, []
    for family, group in groups.items():
        if group:
            predicted, _ = partition.infer(group, params, settings)
            labels.update({ep: (family, f"{family}-{model}") for ep, (_, model) in predicted.items()})
            families.append(sorted(group))
    if unknown:
        predicted, details = partition.infer(unknown, params, settings)
        labels.update({ep: (f"unknown-{family}", f"unknown-{model}")
                       for ep, (family, model) in predicted.items()})
        families.extend(details["families"])
    return labels, {"families": families, "identity_anchors": {ep: tag for ep, tag in identities.items() if tag}}
