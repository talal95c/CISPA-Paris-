"""Reference-only reasoning experiment. No client for scored endpoints or submissions."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from urllib import error, request


ROOT = Path(__file__).resolve().parent
REFS = tuple(f"ref-{i:02}" for i in range(1, 10))
BASE = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
VARIANTS = tuple(ref for ref in REFS if ref not in BASE)
URL = "http://34.147.229.55/33-model-equality/v1/chat/completions"
SUFFIX = "\nReply with just the answer as one integer in digits, without units or explanation."
TOKENS = 64


def bank():
    items = json.loads((ROOT / "bank.json").read_text())
    if (len(items) != 4 or len({it["id"] for it in items}) != 4
            or any(not str(it["gold"]).isdigit() or it["a"] == it["b"] for it in items)):
        raise ValueError("Invalid four-pair bank")
    return {it["id"]: it for it in items}


def jobs(items, block):
    return list(product((block,), items, ("a", "b"), (0, 1), REFS))


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, *args):
        return None


def query(ref, prompt, key):
    if ref not in REFS:
        raise ValueError("Only ref-01..ref-09 are allowed")
    body = json.dumps({"model": ref, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": TOKENS}).encode()
    req = request.Request(URL, data=body, headers={"Content-Type": "application/json",
                                                  "X-API-Key": key})
    for attempt in range(3):
        try:
            with request.build_opener(NoRedirect()).open(req, timeout=90) as r:
                choice = json.load(r)["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("Unexpected non-text response")
            return text, choice.get("finish_reason")
        except error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504) or attempt == 2:
                raise
            time.sleep(max(2 ** (attempt + 1), int(exc.headers.get("Retry-After", "0"))))


def connect(path, items):
    db = sqlite3.connect(path)
    stamp = hashlib.sha256(json.dumps([items, SUFFIX, TOKENS, URL], sort_keys=True).encode()).hexdigest()
    with db:
        db.execute("CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY, digest TEXT)")
        db.execute("INSERT OR IGNORE INTO config VALUES(1,?)", (stamp,))
        if db.execute("SELECT digest FROM config WHERE id=1").fetchone() != (stamp,):
            raise ValueError("Protocol mismatch: use a fresh DB")
        db.execute("""CREATE TABLE IF NOT EXISTS responses
            (block INT, item TEXT, form TEXT, rep INT, ref TEXT, answer TEXT NOT NULL, finish TEXT,
             PRIMARY KEY(block,item,form,rep,ref))""")
    return db


def collect(db, items, block, key):
    done = set(db.execute("SELECT block,item,form,rep,ref FROM responses"))
    pending = [j for j in jobs(items, block) if j not in done]
    if block == 2 and not set(jobs(items, 1)).issubset(done):
        raise ValueError("Complete block 1 before block 2")
    random.Random(15340 + block).shuffle(pending)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(query, ref, items[i][f] + SUFFIX, key): (b, i, f, r, ref)
                   for b, i, f, r, ref in pending}
        try:
            for n, future in enumerate(as_completed(futures), start=1):
                text, finish = future.result()
                with db:
                    db.execute("INSERT INTO responses VALUES(?,?,?,?,?,?,?)",
                               (*futures[future], text, finish))
                if n % 24 == 0:
                    print(f"Block {block}: saved {n}/{len(pending)} pending results", flush=True)
        finally:
            for future in futures:
                future.cancel()


def parse(text):
    # Accept a numeric answer or an explicit, complete final answer line. Do not
    # guess from numbers in a rationale or silently treat a non-answer as wrong.
    lines = text.strip().splitlines()
    if not lines:
        return None
    line = lines[-1].strip().replace("**", "").strip("`").strip()
    m = re.fullmatch(r"(?:(?:the |final )?answer\s*(?:is\s+|:\s*|=\s*))?(-?[0-9]+)[.!]?", line, re.I)
    return int(m[1]) if m else None


def stable(values):
    counts = Counter(v for v in values if v is not None)
    return next((k for k, n in counts.items() if n >= 3), None)


def analyze(db, items):
    rows = list(db.execute("SELECT block,item,form,rep,ref,answer,finish FROM responses"))
    if {r[:5] for r in rows} != set(jobs(items, 1) + jobs(items, 2)):
        raise ValueError("Both complete blocks required")
    values = defaultdict(list)
    for block, i, _, _, ref, answer, finish in rows:
        values[block, i, ref].append(None if finish == "length" else parse(answer))
    summary = []
    for i, item in items.items():
        trials = []
        gold = int(item["gold"])
        for block in (1, 2):
            modes = {ref: stable(values[block, i, ref]) for ref in REFS}
            base = modes[BASE[0]]
            aligned = base is not None and all(modes[r] == base for r in BASE)
            candidates = {r: [base, modes[r]] for r in VARIANTS
                          if aligned and modes[r] is not None and (base == gold) != (modes[r] == gold)}
            trials.append({"modes": modes, "base_aligned": aligned, "candidates": candidates})
        confirmed = [r for r, pair in trials[0]["candidates"].items() if
                     pair == trials[1]["candidates"].get(r)]
        summary.append({"item": i, "trials": trials, "confirmed": confirmed})
    return {"responses": len(rows), "usable": sum(r[6] != "length" and parse(r[5]) is not None for r in rows),
            "truncated": sum(r[6] == "length" for r in rows), "items": summary}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db", type=Path)
    ap.add_argument("--collect-block", type=int, choices=(1, 2))
    args = ap.parse_args()
    items = bank()
    with closing(connect(args.db, items)) as db:
        if args.collect_block:
            collect(db, items, args.collect_block, os.environ["CISPA_KEY"])
        else:
            print(json.dumps(analyze(db, items), indent=2))


if __name__ == "__main__":
    main()
