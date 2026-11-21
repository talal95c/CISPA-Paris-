"""Reference-only reasoning experiment. No client for scored endpoints or submissions."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import closing
from email.utils import parsedate_to_datetime
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import random
import re
import sqlite3
import time
from urllib import error, parse as urlparse, request


ROOT = Path(__file__).resolve().parent
REFS = tuple(f"ref-{i:02}" for i in range(1, 10))
BASE = ("ref-01", "ref-03", "ref-05", "ref-06", "ref-08")
VARIANTS = tuple(ref for ref in REFS if ref not in BASE)
SUFFIX = "\nReply with just the answer as one integer in digits, without units or explanation."
TOKENS = 64


def api_url():
    origin = os.environ.get("CISPA_HTTPS_ORIGIN", "https://34.147.229.55")
    u = urlparse.urlsplit(origin)
    if (u.scheme != "https" or not u.hostname or u.username is not None
            or u.path not in ("", "/") or u.query or u.fragment):
        raise ValueError("Set CISPA_HTTPS_ORIGIN to the organizers' verified HTTPS origin")
    return origin.rstrip("/") + "/33-model-equality/v1/chat/completions"


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


def retry_seconds(header, now):
    if not header:
        return 0
    if header.isascii() and header.isdigit():
        return int(header)
    try:
        return max(0, parsedate_to_datetime(header).timestamp() - now)
    except (ValueError, TypeError, OverflowError):
        return 0


def query(ref, prompt, key):
    if ref not in REFS:
        raise ValueError("Only ref-01..ref-09 are allowed")
    body = json.dumps({"model": ref, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": TOKENS}).encode()
    req = request.Request(api_url(), data=body, headers={"Content-Type": "application/json",
                                                  "X-API-Key": key})
    for attempt in range(8):
        try:
            with request.build_opener(NoRedirect()).open(req, timeout=90) as r:
                choice = json.load(r)["choices"][0]
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("Unexpected non-text response")
            return text, choice.get("finish_reason")
        except error.HTTPError as exc:
            if exc.code not in (429, 502, 503, 504) or attempt == 7:
                raise
            delay = max(min(60, 2 ** (attempt + 1)), retry_seconds(exc.headers.get("Retry-After"), time.time()))
            print(f"{ref}: HTTP {exc.code}; retry after {delay}s", flush=True)
            time.sleep(delay)


def connect(path, items):
    db = sqlite3.connect(path)
    stamp = hashlib.sha256(json.dumps([items, SUFFIX, TOKENS, api_url()], sort_keys=True).encode()).hexdigest()
    with db:
        db.execute("CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY, digest TEXT)")
        db.execute("INSERT OR IGNORE INTO config VALUES(1,?)", (stamp,))
        if db.execute("SELECT digest FROM config WHERE id=1").fetchone() != (stamp,):
            raise ValueError("Protocol mismatch: use a fresh DB")
        db.execute("""CREATE TABLE IF NOT EXISTS responses
            (block INT, item TEXT, form TEXT, rep INT, ref TEXT, answer TEXT NOT NULL, finish TEXT,
             PRIMARY KEY(block,item,form,rep,ref))""")
    return db


def collect(db, items, block, key, workers=1):
    if workers not in range(1, 4):
        raise ValueError("Use 1 to 3 workers")
    done = set(db.execute("SELECT block,item,form,rep,ref FROM responses"))
    pending = [j for j in jobs(items, block) if j not in done]
    if block == 2 and not set(jobs(items, 1)).issubset(done):
        raise ValueError("Complete block 1 before block 2")
    random.Random(15340 + block).shuffle(pending)
    stream = iter(pending)
    failed = None
    n = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        while True:
            while not failed and len(futures) < workers and (j := next(stream, None)):
                _, i, f, _, ref = j
                futures[pool.submit(query, ref, items[i][f] + SUFFIX, key)] = j
            if not futures:
                break
            for future in wait(futures, return_when=FIRST_COMPLETED).done:
                j = futures.pop(future)
                try:
                    text, finish = future.result()
                except Exception as exc:
                    failed = exc
                    continue
                with db:
                    db.execute("INSERT INTO responses VALUES(?,?,?,?,?,?,?)",
                               (*j, text, finish))
                n += 1
                if n % 24 == 0:
                    print(f"Block {block}: saved {n}/{len(pending)} pending results", flush=True)
    if failed:
        raise failed


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
    ap.add_argument("--workers", type=int, default=1, choices=(1, 2, 3))
    args = ap.parse_args()
    items = bank()
    with closing(connect(args.db, items)) as db:
        if args.collect_block:
            collect(db, items, args.collect_block, os.environ["CISPA_KEY"], args.workers)
        else:
            print(json.dumps(analyze(db, items), indent=2))


if __name__ == "__main__":
    main()
