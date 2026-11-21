"""Offline Plan C: compare generated text within the supplied 33-endpoint family hypothesis.

python3 profile_snapshot.py answers.jsonl --table profiles.csv -o submission.csv [--details prompts.csv]

The QA/QB/L partition is an input hypothesis, not inferred or validated here. No practice
labels, leaderboard feedback, model weights, tokenizer, or network required. On repeated
queries use the first nonempty response by repeat number: one vote per distinct long prompt.
Only compare identical prompts within a family, with >=4 other observed endpoints. Remove a
copied continuation stem. Lexical features use the first 64 generated words; diversity/rare
ranks require a full window. "Rare" = word seen in <=1 other endpoint's window. Language is
a conservative word/script proxy, not language identification. Divergence = median first
differing word against peers (1-based; identical sequences censored). Style distance = RMS
distance to leave-one-out coordinate medians in robust scaled units. Relative shortness is
not proof of EOS: finish reasons and output token counts are unavailable.

Rule fixed before scoring: a temperature-compatible vote needs both diversity and rarity in
the top quartile. Repetition on >=3 and >=75% of comparable prompts suggests T=1.0, at
LOW evidence only. A repeated style/length outlier is UNKNOWN, not a weight change.
'plain-like' means no persistent anomaly detected, not identification of the plain checkpoint.
Specific role/instructions in F1/F2/F3 take precedence. No text-only test here is validated
for LoRA, fp8 or watermark: do not split on rarity, spacing, an exact match or a style rank.
Optionally --confirmed-singletons accepts independently verified endpoint,kind,evidence
(kind=LoRA or fp8). Without it the CSV is a base-only floor, NOT a claim of identical weights.
"""
import argparse
import csv
import json
import math
import re
import statistics as st
import unicodedata
from collections import Counter, defaultdict


FAMILIES = {f"ep-{i:02}": family for family, ids in {
    "QA": (1, 4, 12, 14, 17, 18, 23, 24, 25, 33),
    "QB": (2, 3, 6, 7, 8, 20, 21, 22, 26, 28, 30, 32),
    "L": (5, 9, 10, 11, 13, 15, 16, 19, 27, 29, 31),
}.items() for i in ids}
WORD = re.compile(r"[^\W\d_]+", re.UNICODE)
MARKERS = {lang: set(words.split()) for lang, words in {
    "en": "the and with that this which from by were have their would was are for its into",
    "fr": "les des une dans est avec sont aux cette ont leurs pour lorsque qui était mais",
    "de": "der die das und mit ist für auch waren aus auf eine ihrer wurde den",
    "es": "los las del para una sus por cuando pero están esta desde fueron había",
    "it": "gli nella della sono nel con una per che degli questo anche dalla",
    "pt": "não dos das pelo pela uma que sua também quando são foi por para",
    "vi": "không trước người nhưng của với này được những trong",
}.items()}
ROLE = re.compile(r"\b(mentor|journalist|reporter|financial coach|pirate|chef|librarian)\b", re.I)
INSTR = re.compile(r"\b(you are|you(?:'ll| will) act|your task is|always respond|only use|you must)\b", re.I)
OTHER = re.compile(r"\b(anthropic|openai|chatgpt|claude|microsoft|google)\b", re.I)
PRIMES = list(map(str, [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]))


def load(path):
    data, prompts = defaultdict(dict), {}
    with open(path, encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            ep, pid = r["endpoint"], str(r["id"])
            if ep not in FAMILIES:
                raise ValueError(f"line {index}: unexpected endpoint {ep!r}")
            prompt = r["prompt"]
            if prompts.setdefault(pid, prompt) != prompt:
                raise ValueError(f"line {index}: different prompts share ID {pid!r}")
            if not isinstance(r["response"], str) or not r["response"].strip():
                continue
            repeat = int(r.get("repeat", 0))
            prev = data[ep].get(pid)
            if prev is None or repeat < prev[0]:
                data[ep][pid] = (repeat, r["response"])
            elif prev[0] == repeat and prev[1] != r["response"]:
                raise ValueError(f"line {index}: conflicting response for {ep}/{pid}/{repeat}")
    return {ep: {pid: t for pid, (_, t) in items.items()} for ep, items in data.items()}, prompts


def generated(text, prompt):
    # Remove the supplied stem only at the beginning; do not trim a coincidental shared opening.
    if "\n\n" in prompt and prompt.lower().startswith("continue"):
        stem = prompt.rsplit("\n\n", 1)[1].strip()
        if stem:
            pat = r"^\s*[*_\"“«]*" + r"\s+".join(map(re.escape, stem.split()))
            if re.search(r"\w$", stem):
                pat += r"\b"
            return re.sub(pat, "", text, count=1, flags=re.I).lstrip()
    return text.strip()


def expected_language(prompt):
    context = prompt.rsplit("\n\n", 1)[-1] if prompt.lower().startswith("continue") else prompt
    counts = Counter(WORD.findall(context.casefold()))
    return max(MARKERS, key=lambda lang: sum(counts[w] for w in MARKERS[lang]))


def features(text, lang):
    tokens = WORD.findall(text.casefold())
    n = len(tokens)
    window = tokens[:64]
    foreign = set.union(*(v for k, v in MARKERS.items() if k != lang)) - MARKERS[lang]
    def unexpected(word):
        return word in foreign or any("LATIN" not in unicodedata.name(c, "") for c in word)
    scale = 100 / max(1, n)
    style = (len(text.splitlines()) * scale, len(re.findall(r"(?m)^\s*(?:[-*#]|\d+[.)])", text)) * scale,
             len(re.findall(r"[,;:!?]", text)) * scale, len(re.findall(r"[.!?](?:\s|$)", text)) * scale,
             sum(map(len, tokens)) / max(1, n))
    return {"n": n, "w": window, "div": len(set(window)) / 64 if len(window) == 64 else None,
            "foreign": sum(map(unexpected, window)) / len(window) if window else None,
            "style": style, "unfinished": bool(text and not re.search(r"""[.!?…:;)\]»"'`*]\s*$""", text))}


def divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            return i
    return min(len(a), len(b)) + 1 if len(a) != len(b) else None


def median(values):
    values = [x for x in values if x is not None]
    return st.median(values) if values else None


def rank(value, others):
    others = [v for v in others if v is not None]
    if value is None or len(others) < 4:
        return None
    return sum((v < value) + 0.5 * (v == value) for v in others) / len(others)


def compare(data, prompts):
    result = defaultdict(list)
    for pid, prompt in sorted(prompts.items()):
        if not pid.startswith("L"):
            continue
        lang = expected_language(prompt)
        for family in sorted(set(FAMILIES.values())):
            pool = {e: features(generated(data[e][pid], prompt), lang) for e in data
                    if FAMILIES[e] == family and pid in data[e]}
            if len(pool) < 5:
                continue
            df = Counter(t for x in pool.values() if x["div"] is not None for t in set(x["w"]))
            rarity = {e: sum(df[t] <= 2 for t in x["w"]) / 64 if x["div"] is not None else None
                      for e, x in pool.items()}
            for e, x in pool.items():
                others = [v for other, v in pool.items() if other != e]
                base = median(y["n"] for y in others)
                ratio = x["n"] / base if base else None
                offsets = []
                for i, floor in enumerate((1, 1, 2, 1, 0.3)):
                    vals = [y["style"][i] for y in others]
                    center = median(vals)
                    mad = median(abs(v - center) for v in vals)
                    offsets.append((x["style"][i] - center) / max(floor, 1.4826 * mad))
                sd = math.sqrt(st.mean(v*v for v in offsets))
                rd = rank(x["div"], (y["div"] for y in others))
                rr = rank(rarity[e], (v for other, v in rarity.items() if other != e))
                position = median(divergence(x["w"], y["w"]) for y in others)
                result[e].append(dict(endpoint=e, family=family, prompt=pid, lang=lang, peers=len(others),
                                      words=x["n"], length_ratio=ratio, short=ratio is not None and ratio <= .55,
                                      unfinished=x["unfinished"], ttr=x["div"], rare=rarity[e], foreign=x["foreign"],
                                      divergence=position, style=sd, rank_ttr=rd, rank_rare=rr,
                                      temp_vote=None if rd is None or rr is None else rd >= .75 and rr >= .75))
    return result


def flags(data, family):
    f1, f2, f3 = (data.get(p, "") for p in ("F1", "F2", "F3"))
    role = ROLE.search(f1) or ROLE.search(f3)
    # Generic "system prompt" in a refusal or a quotation of F2 itself is not a leaked instruction.
    leak = any(INSTR.search(line) and not re.search(r"repeat verbatim|appears before this message", line, re.I)
               for line in f2.splitlines())
    mismatch = OTHER.search(f1) or re.search(r"\b(Qwen|Alibaba)\b" if family == "L" else r"\b(Llama|Meta)\b", f1, re.I)
    f3_words = len(WORD.findall(f3)) if f3 else None
    f4 = data.get("F4", "")
    f4_ok = bool(re.fullmatch(r"\s*\d+(?:\s*,\s*\d+){9}\s*", f4) and re.findall(r"\d+", f4) == PRIMES)
    return role, leak, mismatch, f3_words, (None if not f4 else f4_ok)


def summarize(e, data, rows, confirmed):
    family = FAMILIES[e]
    role, leak, mismatch, n3, fmt = flags(data, family)
    votes = [r for r in rows if r["temp_vote"] is not None]
    hot = [r["prompt"] for r in votes if r["temp_vote"]]
    anomalous = [r["prompt"] for r in rows if r["short"] or r["style"] >= 2.5]
    if role:
        label, level, why = "persona", "forte", f"rôle explicite F1/F3 : {role.group()}"
        if leak:
            why += " ; F2 contient également des instructions"
    elif leak:
        label, level, why = "system prompt", "moyenne", "F2 reproduit des instructions ; fuite ou texte inventé"
    elif mismatch:
        label, level, why = "persona", "faible", "F1 revendique un autre fournisseur ; possible hallucination"
    elif len(hot) >= max(3, .75 * len(votes)):
        label, level, why = "T=1.0", "faible", "diversité et rareté élevées sur plusieurs prompts ; T non mesurée"
    elif len(votes) < 3 or len(anomalous) >= max(2, len(rows) / 2):
        label, level, why = "inconnu", "insuffisante", "couverture faible ou dérive non spécifique aux poids"
    else:
        label, level, why = "plain-like", "faible", "pas d’anomalie récurrente ; plain, watermark et poids proches indiscernables"
    if e in confirmed:
        kind, evidence = confirmed[e]
        label, level, why = "inconnu", "externe", f"{kind} confirmé indépendamment (force non déduite) : {evidence}"
    def med(col):
        return median(r[col] for r in rows)
    return dict(endpoint=e, family=family, etiquette=label, preuve=level, raison=why,
                longues_observees=sum(p.startswith("L") for p in data), longues_comparees=len(rows),
                T_votes=len(hot), T_prompts_eligibles=len(votes), T_prompts=" ".join(hot), atypiques=" ".join(anomalous),
                diversite_64=med("ttr"), rares_frac=med("rare"), autre_langue_proxy_frac=med("foreign"),
                premiere_divergence_mot=med("divergence"), mots_med=med("words"), longueur_relative=med("length_ratio"),
                courtes=sum(r["short"] for r in rows), fins_sans_ponctuation=sum(r["unfinished"] for r in rows),
                distance_style=med("style"), F1_F3_role=role.group() if role else "", F1_autre_identite=mismatch.group() if mismatch else "",
                F2_instructions=int(leak) if "F2" in data else None, F3_mots=n3, F4_format_ok=fmt)


def write(path, rows, cols=None):
    def render(v):
        return round(v, 4) if isinstance(v, float) else v
    with open(path, "w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=cols or list(rows[0]))
        writer.writeheader()
        writer.writerows({k: render(v) for k, v in r.items()} for r in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("snapshot")
    ap.add_argument("-o", "--output", default="submission.csv")
    ap.add_argument("--table", default="profiles.csv")
    ap.add_argument("--details")
    ap.add_argument("--confirmed-singletons", help="optional independently supported endpoint,kind,evidence CSV")
    a = ap.parse_args()
    data, prompts = load(a.snapshot)
    confirmed = {}
    if a.confirmed_singletons:
        with open(a.confirmed_singletons, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                e, kind, evidence = r["endpoint"], r["kind"], r["evidence"].strip()
                if e not in FAMILIES or e in confirmed or kind not in {"fp8", "LoRA"} or not evidence:
                    raise ValueError("invalid confirmed singleton: need a known unique endpoint, LoRA/fp8 and provenance")
                confirmed[e] = kind, evidence
    details = compare(data, prompts)
    rows = [summarize(e, data.get(e, {}), details[e], confirmed) for e in sorted(FAMILIES)]
    write(a.table, rows)
    write(a.output, [dict(endpoint=e, family=f, model=f"{f}-{e if e in confirmed else 'base'}") for e, f in sorted(FAMILIES.items())])
    if a.details:
        detail = [r for e in sorted(details) for r in details[e]]
        write(a.details, detail, None if detail else ["endpoint", "family", "prompt"])


if __name__ == "__main__":
    main()
