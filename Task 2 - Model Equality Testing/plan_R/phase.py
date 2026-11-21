"""Phase 0 diagnostics on local generations; writes measured gate decisions."""
import argparse
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np

from features import canonical_choice, char_signature, complete_format, cosine, normalize, parse_burst, read_rows, signatures
from simulate import phase_slots, replay_slots


def panel(rows, slots):
    ids = {s.id: s for s in slots}
    result = defaultdict(dict)
    for row in rows:
        slot = ids[row["id"]]
        parsed = parse_burst(row["response"], slot.items)
        result[slot.id][row["repeat"]] = [
            canonical_choice(item, parsed[item.tag]) if item.tag in parsed else np.nan for item in slot.items
        ]
    return {pid: np.asarray([values[k] for k in sorted(values)], dtype=float) for pid, values in result.items()}


def effectiveness(panels):
    independent, actual, complete = 0.0, 0.0, 0
    total = sum(len(a) for a in panels.values())
    for a in panels.values():
        a = a[np.isfinite(a).all(axis=1)]
        complete += len(a)
        if len(a) < 2:
            continue
        independent += (len(a)-1)*a.var(axis=0, ddof=1).sum()
        actual += (len(a)-1)*a.sum(axis=1).var(ddof=1)
    return (min(1.0, independent/actual) * complete/total if actual and total else None, complete)


def null_baseline(panels):
    within, available = [], 0
    for data in panels.values():
        a, b = data[::2], data[1::2]
        for x, y in zip(a, b):
            both = np.isfinite(x) & np.isfinite(y)
            if both.any():
                within.append(float(np.mean(x[both] != y[both])))
                available += int(both.sum())
    return {"comparisons_with_items": len(within), "matched_items": available,
            "mean": float(np.mean(within)) if within else None,
            "q95": float(np.quantile(within, .95)) if within else None}


def free_text_null(rows):
    by_prompt = defaultdict(dict)
    for row in rows:
        by_prompt[row["id"]][row["repeat"]] = normalize(char_signature(row["response"]))
    differences = []
    for choices in by_prompt.values():
        for rep in sorted(choices):
            if rep % 2 == 0 and rep+1 in choices and choices[rep] and choices[rep+1]:
                differences.append(1-cosine(choices[rep], choices[rep+1]))
    return {"text_pairs": len(differences),
            "text_mean": float(np.mean(differences)) if differences else None,
            "text_q95": float(np.quantile(differences, .95)) if differences else None}


def watermark_gate(plain, marked, draws=500):
    rng = np.random.default_rng(3807)
    matrix = []
    for pid in sorted(plain.keys() & marked.keys()):
        a, b = plain[pid], marked[pid]
        a = a[np.isfinite(a).all(axis=1)]
        b = b[np.isfinite(b).all(axis=1)]
        n = min(len(a)//2, len(b))
        if n >= 3:
            matrix.append((a[:n], a[n:2*n], b[:n]))
    if not matrix:
        return {"verdict": "non mesurable", "variance": None, "null_q95": None, "items": 0}
    deltas = np.concatenate([c.mean(axis=0) - a.mean(axis=0) for a, _, c in matrix])
    null = []
    for _ in range(draws):
        draw = []
        for a, b, _ in matrix:
            sign = rng.choice([-1, 1], size=(len(a), 1))
            draw.extend(((a-b)*sign).mean(axis=0))
        null.append(np.var(draw, ddof=1))
    value, q95 = float(np.var(deltas, ddof=1)), float(np.quantile(null, .95))
    return {"variance": value, "null_q95": q95, "items": len(deltas),
            "verdict": "go" if value <= q95 else "no-go"}


def auc(positives, negatives):
    values = [float(p > n) + .5*float(p == n) for p, n in product(positives, negatives)]
    return float(np.mean(values)) if values else None


def temperature_result(temp):
    positive = temp["temp"]
    same_names = ("plain", "system", "persona", "watermark")
    negative_names = ("fp8", "lora03", "lora07", "lora15")
    same = [v for name in same_names for _, v in temp[name]]
    variants = [entry for name in negative_names for entry in temp[name]]
    score = auc([v for _, v in positive], same)
    alarms = []
    for block, value in variants:
        a = [p for held, p in positive if block != held]
        b = [p for name in same_names for held, p in temp[name] if block != held]
        if a and b:
            threshold = (float(np.median(a)) + float(np.median(b))) / 2
            alarms.append(value >= threshold)
    fpr = float(np.mean(alarms)) if alarms else None
    return {"auc": score, "fpr": fpr, "n_positives": len(positive), "n_same": len(same),
            "n_variants": len(variants), "n_fpr": len(alarms),
            "verdict": ("non mesurable" if score is None or fpr is None else
                        "go" if score >= .9 and fpr <= .1 else "no-go")}


def phase(rows, size):
    phase_runs, replay_runs = defaultdict(dict), defaultdict(list)
    for row in rows:
        if row["event"] != "answer":
            continue
        block = row["block"]
        if isinstance(block, int):
            replay_runs[row.get("heldout", False), block].append(row)
        else:
            phase_runs[block][row["endpoint"], row["id"], row["repeat"]] = row
    result = {}
    for size_phase in (20, 12):
        slots = phase_slots(size=size_phase)
        by_id = {s.id: s for s in slots}
        data = defaultdict(list)
        for row in phase_runs[f"phase{size_phase}"].values():
            data[row["variant"]].append(row)
        plain = panel(data["plain"], slots)
        marked = panel(data["watermark"], slots)
        efficiency, useful = effectiveness(plain)
        malformed = {v: (1 - float(np.mean([complete_format(r["response"], by_id[r["id"]].items) for r in data[v]]))
                         if data[v] else None) for v in ("system", "persona")}
        format_ok = all(x is not None and x <= .20 for x in malformed.values())
        parsed_rate = [float(np.isfinite(a).mean()) for a in plain.values()]
        exact_pairs = {}
        for variant in ("plain", "greedy"):
            by_prompt = defaultdict(list)
            for r in data[variant]:
                by_prompt[r["id"]].append(r["response"].strip())
            exact_pairs[variant] = [bool(a and a == b) for choices in by_prompt.values()
                                    for a, b in zip(choices[::2], choices[1::2])]
        baseline = null_baseline(plain)
        text_null = free_text_null(data["plain"])
        greedy_identity = (bool(exact_pairs["greedy"]) and bool(exact_pairs["plain"]) and
                           all(exact_pairs["greedy"]) and not any(exact_pairs["plain"]))
        result[str(size_phase)] = {
            "B2": {"n": len(data["plain"]), **baseline, **text_null,
                   "verdict": ("mesurée" if baseline["matched_items"] else
                               "texte seulement" if text_null["text_pairs"] else "non mesurable")},
            "B2prime": {**{v: {"identical": sum(pairs), "n": len(pairs)} for v, pairs in exact_pairs.items()},
                        "verdict": "go" if greedy_identity else "no-go"},
            "B3": {"efficiency": efficiency, "plain_complete": useful, "plain_n": len(data["plain"]),
                   "plain_item_coverage": float(np.mean(parsed_rate)) if parsed_rate else None,
                   "off_format": malformed, "off_format_n": {v: len(data[v]) for v in malformed},
                   "verdict": ("go" if format_ok and efficiency is not None and efficiency >= .30 else "no-go")},
            "B4": watermark_gate(plain, marked),
        }
    flags, temp, shape, flag_errors = defaultdict(list), defaultdict(list), defaultdict(list), Counter()
    negative_names = ("fp8", "lora03", "lora07", "lora15")
    for (heldout, run), data in replay_runs.items():
        variants = {r["endpoint"]: r["variant"] for r in data}
        for ep, sig in signatures(data, replay_slots(run, size, heldout=heldout)).items():
            name = variants[ep]
            flag = sig.flags["strength"] >= 1.5
            flags[name].append(flag)
            flag_errors[name] += flag
            if sig.repeat["disagreement"] is not None:
                temp[name].append(((heldout, run), sig.repeat["disagreement"]))
            if sig.repeat["shape"] is not None:
                shape[name].append(((heldout, run), sig.repeat["shape"]))
    rates = {name: {"rate": flag_errors[name]/len(points), "n": len(points)} for name, points in flags.items()}
    flag_go = (all(rates.get(v, {"rate": 0})["rate"] >= .9 for v in ("system", "persona")) and
               all(rates.get(v, {"rate": 1})["rate"] <= .05 for v in negative_names))
    result["flags"] = {"by_variant": rates, "verdict": "go" if flag_go else "no-go"}
    result["B7"] = {**temperature_result(temp), "shape": temperature_result(shape)}
    rates = [result[str(s)]["B3"]["efficiency"] for s in (20, 12)]
    result["C"] = {"burst_fallback_to_R": None if rates[1] is None else rates[1] < .25,
                   "recommendation": "20 items" if result["20"]["B3"]["verdict"] == "go" else "12 items"}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    journals = [read_rows(journal) for journal in args.journal]
    if any(not journal or journal[0].get("event") != "simulation" for journal in journals):
        parser.error("all journals must start with the simulation header")
    if any(journal[0]["items"] != journals[0][0]["items"] for journal in journals):
        parser.error("journals must share the replay item count")
    rows = [dict(row, heldout=journal[0].get("heldout", False)) for journal in journals for row in journal[1:]]
    result = phase(rows, journals[0][0]["items"])
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
