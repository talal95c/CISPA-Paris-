"""answers.jsonl (partial OK) -> submission.csv (endpoint,family,model). Offline only, never contacts an endpoint.

Usage: python cluster.py answers.jsonl [-o submission.csv] [--split-threshold 0.25] [--endpoints 33]

Floor rule (plan R):
  family : identity keywords in F1 (Qwen/Alibaba, Llama/Meta, ...); endpoints without a keyword use unambiguous
           exact long-answer matches, then character 4-gram distance, else the largest group of anchors. A keyword group
           is split by F4 comma spacing when each format has >= 3 endpoints (A: no spaces, B: spaces).
           Otherwise, split when the average-linkage gap on long-answer profiles exceeds --split-threshold.
  model  : endpoints flagged as disguised (persona: F1 off-identity; system prompt: F2 leaks instructions) share the
           family's S label; long answers duplicated verbatim also merge; every other endpoint stays a singleton.
Diagnostics go to stderr.
"""
import argparse
import collections
import csv
import json
import math
import re
import sys

IDENTITIES = {
    "qwen": ["qwen", "alibaba", "tongyi"],
    "llama": ["llama", "meta ai", "meta platforms", "by meta", "meta."],
    "mistral": ["mistral"],
    "gemma": ["gemma", "google deepmind", "by google"],
    "phi": ["phi-", "microsoft"],
    "smollm": ["smollm", "hugging face", "huggingface"],
    "olmo": ["olmo", "allen institute", "ai2"],
}
NO_CONTEXT = re.compile(r"\b(no (text|previous|prior|instructions|messages?)|there is none|none\b|nothing|this is the (first|beginning)|"
                        r"no preceding|no earlier|start of (our|the) conversation|first message)", re.I)
ROLE = re.compile(r"\b(journalist|mentor|coach|tutor|chef|pirate|advisor|adviser|consultant|specialist|expert|guide|"
                  r"companion|concierge|librarian|teacher|agent named|my name is (?!qwen|llama))\b", re.I)
LEAK = re.compile(r"\b(you are|you'll act|you will act|your (role|task|name) is|act as|always respond|you must|only use|"
                  r"do not mention|system prompt)\b", re.I)
OWN_F2 = re.compile(r"repeat verbatim|appears before this message|forgot to include|"
                    r"(can(no|')t|unable to|not able to|won't) (share|reveal|disclose|repeat|provide)", re.I)


def load(path):
    data = collections.defaultdict(lambda: collections.defaultdict(list))
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            data[r["endpoint"]][str(r["id"])].append(r.get("response") or "")
    return data


def identity(text):
    t = text.lower()
    hits = [k for k, words in IDENTITIES.items() if any(w in t for w in words)]
    return hits[0] if len(hits) == 1 else None


def profile(answers):
    """Normalised char 4-gram counts over all long answers (ids starting with L)."""
    c = collections.Counter()
    for pid, resp in answers.items():
        if pid.startswith("L"):
            for r in resp:
                s = re.sub(r"\s+", " ", r.lower())
                c.update(s[i:i + 4] for i in range(len(s) - 3))
    n = math.sqrt(sum(v * v for v in c.values()))
    return {k: v / n for k, v in c.items()} if n else None


def cos_dist(a, b):
    if len(a) > len(b):
        a, b = b, a
    return 1 - sum(v * b.get(k, 0.0) for k, v in a.items())


def common_prefix(a, b):
    """Mean number of identical leading words over the long prompts both endpoints answered."""
    vals = []
    for pid in set(a) & set(b):
        if pid.startswith("L") and a[pid] and b[pid]:
            wa, wb = a[pid][0].split(), b[pid][0].split()
            n = 0
            while n < min(len(wa), len(wb)) and wa[n] == wb[n]:
                n += 1
            vals.append(n)
    return sum(vals) / len(vals) if vals else None


def long_keys(answers):
    return {(pid, r) for pid, resp in answers.items() if pid.startswith("L") for r in resp if len(r) > 200}


def split_two(eps, prof, thr):
    """Average-linkage agglomeration down to 2 clusters; split only if the last merge distance > thr."""
    have = [e for e in eps if prof.get(e)]
    if len(have) < 4:
        return [eps], None
    cl = [[e] for e in have]
    d = {}
    def dist(x, y):
        return sum(d.setdefault((min(a, b), max(a, b)), cos_dist(prof[a], prof[b])) for a in x for b in y) / (len(x) * len(y))
    while len(cl) > 2:
        i, j = min(((i, j) for i in range(len(cl)) for j in range(i + 1, len(cl))), key=lambda p: dist(cl[p[0]], cl[p[1]]))
        cl[i] += cl.pop(j)
    gap = dist(cl[0], cl[1])
    within = max((dist([a], [b]) for c in cl for a in c for b in c if a < b), default=0.0)
    if gap <= thr or min(len(c) for c in cl) < 2:
        return [eps], gap
    missing = [e for e in eps if e not in have]
    cl[max(range(2), key=lambda k: len(cl[k]))] += missing
    print(f"  split: gap={gap:.3f} (max within {within:.3f}) -> {cl}", file=sys.stderr)
    return cl, gap


def f4_signature(answers):
    signatures = set()
    for response in answers.get("F4", []):
        text = response.strip()
        if not re.fullmatch(r"\d+(?:,[ \t]*\d+)+", text):
            return None
        if [int(n) for n in re.findall(r"\d+", text)] != [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]:
            return None
        spaces = re.findall(r",([ \t]*)", text)
        if all(not s for s in spaces):
            signatures.add("compact")
        elif all(spaces):
            signatures.add("spaced")
        else:
            return None
    return next(iter(signatures)) if len(signatures) == 1 else None


def split_f4(name, members, data, enabled=True):
    groups = {s: [] for s in ["compact", "spaced", None]}
    for e in members:
        groups[f4_signature(data[e])].append(e)
    print(f"{name}: F4 " + "; ".join(f"{s or 'unsigned'}={len(p)} {p}" for s, p in groups.items()), file=sys.stderr)
    if not enabled or min(len(groups[s]) for s in ["compact", "spaced"]) < 3:
        return None
    parts = [groups["compact"], groups["spaced"]]
    for e in groups[None]:
        parts[min(range(2), key=lambda i: len(parts[i]))].append(e)
    print(f"  F4 split: {name}A (compact)={parts[0]}; {name}B (spaced)={parts[1]}; L* split skipped", file=sys.stderr)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("answers")
    ap.add_argument("-o", "--output", default="submission.csv")
    ap.add_argument("--split-threshold", type=float, default=0.25)
    ap.add_argument("--endpoints", type=int, default=33)
    ap.add_argument("--no-f4-split", action="store_true", help="disable family splitting by F4 comma spacing")
    ap.add_argument("--prefix-link", type=int, default=0,
                    help="merge two endpoints of a family whose long answers share >= N leading words on average "
                         "(0 = off; risky: near-greedy plain, fp8 and LoRA 0.3 share prefixes too)")
    a = ap.parse_args()

    data = load(a.answers)
    eps = sorted(set(data) | {f"ep-{i:02d}" for i in range(1, a.endpoints + 1)})
    prof = {e: profile(data[e]) for e in eps}
    f1 = {e: " ".join(data[e].get("F1", [])) for e in eps}
    ident = {e: identity(f1[e]) for e in eps}

    groups = collections.defaultdict(list)
    for e in eps:
        if ident[e]:
            groups[ident[e]].append(e)
    if not groups:
        groups["fam"] = eps
    anchors, members_by_name = {}, {}
    tentative_labels = collections.defaultdict(list)
    for name, members in sorted(groups.items()):
        counts = collections.Counter(f4_signature(data[e]) for e in members)
        if not a.no_f4_split and min(counts["compact"], counts["spaced"]) >= 3:
            parts = [members]
        else:
            parts, gap = split_two(members, prof, a.split_threshold)
            print(f"{name}: {len(members)} endpoints, 2-split gap={gap if gap is None else round(gap, 3)}", file=sys.stderr)
        for k, p in enumerate(parts, 1):
            label = f"{name}{k}" if len(parts) > 1 else name
            anchors[label] = p
            members_by_name[label] = p.copy()
            tentative_labels[name].append(label)

    family = {e: name for name, p in members_by_name.items() for e in p}
    exact = collections.defaultdict(set)
    for e, label in family.items():
        for key in long_keys(data[e]):
            exact[key].add(label)
    for e in eps:
        if e in family:
            continue
        matches = {label for key in long_keys(data[e]) for label in exact.get(key, ())}
        if len(matches) == 1:
            label, = matches
        else:
            distances = {}
            if prof[e]:
                for name, refs in anchors.items():
                    ds = [cos_dist(prof[e], prof[m]) for m in refs if prof[m]]
                    if ds:
                        distances[name] = sum(ds) / len(ds)
            label = min(distances, key=distances.get) if distances else max(anchors, key=lambda n: len(anchors[n]))
        family[e] = label
        members_by_name[label].append(e)
    fams = []
    for name, labels in tentative_labels.items():
        members = sorted(e for label in labels for e in members_by_name[label])
        parts = split_f4(name, members, data, enabled=not a.no_f4_split)
        if parts:
            fams.extend(zip((name + "A", name + "B"), parts))
        else:
            fams.extend((label, members_by_name[label]) for label in labels)
    family = {e: name for name, p in fams for e in p}
    flags = {}
    for e in eps:
        why = []
        if f1[e] and ident[e] is None:
            why.append("persona(F1 off-identity)")
        elif ROLE.search(f1[e]):
            why.append("persona(F1 role)")
        f2 = " ".join(data[e].get("F2", []))
        body = re.sub(r"^\W*none\W*", "", f2.strip(), flags=re.I)
        leak, own = LEAK.search(body), OWN_F2.search(body)
        if body and leak and not (own and own.start() < leak.start()):
            why.append("sys(F2 leak)")
        flags[e] = why

    parent = {e: e for e in eps}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for name, p in fams:
        flagged = [e for e in p if flags[e]]
        for e in flagged[1:]:
            parent[find(e)] = find(flagged[0])
    seen = {}
    for e in eps:
        for pid, r in sorted(long_keys(data[e])):
            o = seen.setdefault((family[e], pid, r), e)
            if o != e:
                parent[find(e)] = find(o)
                print(f"  duplicate {pid}: {o} = {e}", file=sys.stderr)

    if a.prefix_link:
        for name, p in fams:
            for i, x in enumerate(p):
                for y in p[i + 1:]:
                    cp = common_prefix(data[x], data[y])
                    if cp is not None and cp >= a.prefix_link:
                        parent[find(y)] = find(x)
                        print(f"  prefix {x} = {y} ({cp:.0f} words)", file=sys.stderr)

    with open(a.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["endpoint", "family", "model"])
        for e in eps:
            root = find(e)
            model = f"{family[e]}-S" if any(flags[x] for x in eps if find(x) == root) else f"{family[e]}-{root}"
            w.writerow([e, family[e], model])
            print(f"{e} fam={family[e]:8s} model={model:12s} flags={','.join(flags[e]) or '-':28s} "
                  f"F1={f1[e][:70]!r}", file=sys.stderr)
    print(f"wrote {a.output} ({len(eps)} endpoints; {sum(1 for e in eps if not data[e])} without answers)", file=sys.stderr)


if __name__ == "__main__":
    main()
