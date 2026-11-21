"""Collect Task 2 completions; the default and --dry-run never open a socket."""
import argparse
import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from protocol import fingerprint, manifest


SERVER = "http://34.147.229.55"
TASK_PATH = "/33-model-equality/v1"


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def request(base, path, key, payload=None):
    headers = {"X-API-Key": key}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = Request(base + path, json.dumps(payload).encode() if payload else None, headers)
    try:
        response = build_opener(NoRedirect()).open(req, timeout=180 if payload else 60)
    except HTTPError as exc:
        response = exc
    with response:
        raw = response.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw)
        except ValueError:
            body = raw
        return response.code, response.headers, body


def retry_delay(raw):
    try:
        return max(0, int(raw))
    except ValueError:
        return max(0, (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds())


def completion(endpoint, prompt):
    return {"model": endpoint, "max_tokens": 256, "messages": [{"role": "user", "content": prompt}]}


def answer_text(body):
    try:
        text = body["choices"][0]["message"]["content"]
        return text if isinstance(text, str) else None
    except (AttributeError, KeyError, TypeError, IndexError):
        return None


def collect(base, key, endpoints, slots, output, wave, scored):
    manifest_id = fingerprint(slots)
    by_id = {slot.id: slot for slot in slots}
    lock = Lock()
    with output.open("a+", encoding="utf-8") as journal:
        fcntl.flock(journal, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal.seek(0)
        rows = {}
        pending = set()
        header = {"event": "manifest", "sha256": manifest_id, "source": base, "scored": scored}

        def log(row):
            with lock:
                journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                journal.flush()
                os.fsync(journal.fileno())

        saved_header = next(journal, None)
        if saved_header is None:
            log(header)
        elif json.loads(saved_header) != header:
            raise ValueError("journal uses a different manifest or source")
        for line in journal:
            row = json.loads(line)
            job = row["endpoint"], row["id"]
            if row["event"] in {"start", "uncertain"}:
                pending.add(job)
            else:
                pending.discard(job)
                if row["event"] in {"answer", "rejected"}:
                    rows[job] = row
        if pending:
            raise RuntimeError(f"unresolved dispatches {sorted(pending)}; reconcile with the team before resuming")

        status, _, body = request(base, "/models", key)
        if status != 200:
            raise RuntimeError(f"GET /models returned HTTP {status}")
        budgets = {model["id"]: model.get("budget") for model in body["data"]}
        for ep in endpoints:
            if ep not in budgets:
                raise ValueError(f"server does not list {ep}")
            if scored:
                used = sum(e == ep and row["event"] == "answer" for (e, _), row in rows.items())
                if budgets[ep]["used"] != used or budgets[ep]["limit"] != 100:
                    raise RuntimeError(f"{ep}: server budget {budgets[ep]} disagrees with {used} saved answers")

        def one(ep, slot):
            job = ep, slot.id
            if job in rows:
                return
            probe = slot
            if slot.id.startswith("reserve-"):
                with lock:
                    snapshot = tuple(rows.items())
                replaced = {r["probe_id"] for (e, _), r in snapshot if e == ep and
                            r["id"].startswith("reserve-") and r["event"] == "answer" and r["response"]}
                failures = [s for s in slots if not s.id.startswith("reserve-") and s.id not in replaced
                            and (ep, s.id) in rows and not rows[ep, s.id].get("response")]
                if failures:
                    probe = failures[0]
            payload = completion(ep, probe.prompt)
            entry = {"endpoint": ep, "id": slot.id, "probe_id": probe.id, "kind": by_id[probe.id].kind,
                     "repeat": 0, "prompt": probe.prompt, "request": payload}
            for _ in range(8):
                log(dict(entry, event="start"))
                # Ambiguous failures leave an unresolved start; retrying could consume another answer.
                status, headers, body = request(base, "/chat/completions", key, payload)
                record = dict(entry, status=status, body=body)
                if status == 429 and headers.get("Retry-After"):
                    log(dict(record, event="busy"))
                    time.sleep(retry_delay(headers["Retry-After"]))
                    continue
                if status == 429:
                    log(dict(record, event="exhausted"))
                    raise RuntimeError(f"{ep}: exhausted before the manifest finished")
                if status == 200:
                    record.update(event="answer", response=answer_text(body))
                    log(record)
                    with lock:
                        rows[job] = record
                    if scored:
                        expected = budgets[ep]["used"] + 1
                        current = body.get("budget") if isinstance(body, dict) else None
                        if not isinstance(current, dict) or (current.get("used"), current.get("limit")) != (expected, 100):
                            raise RuntimeError(f"{ep}: expected budget {expected}/100, received {current}")
                        budgets[ep] = current
                    return
                if status in {400, 413, 422}:
                    record.update(event="rejected")
                    log(record)
                    with lock:
                        rows[job] = record
                    return
                log(dict(record, event="uncertain"))
                raise RuntimeError(f"{ep}: unexpected HTTP {status}; reconcile before resuming")
            raise RuntimeError(f"{ep}: busy after eight attempts; resume later")

        # A barrier between slots keeps requests round-robin with at most one in flight per endpoint.
        with ThreadPoolExecutor(max_workers=4) as pool:
            for slot in slots:
                if wave == "all" or slot.wave <= int(wave):
                    list(pool.map(lambda ep: one(ep, slot), endpoints))
        return sum(up["event"] == "answer" for up in rows.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--mock", help="loopback server, e.g. http://127.0.0.1:8080")
    modes.add_argument("--live", action="store_true")
    p.add_argument("--confirm-scored", action="store_true")
    p.add_argument("--target", choices=("scored", "refs"), default="scored")
    p.add_argument("--wave", choices=("1", "2", "all"), default="all")
    p.add_argument("--items", type=int, choices=[12, *range(20, 26)], default=20)
    p.add_argument("--output", type=Path, default=Path("answers.jsonl"))
    args = p.parse_args()
    scored = args.target == "scored"
    endpoints = [f"{'ep' if scored else 'ref'}-{n+1:02d}" for n in range(33 if scored else 9)]
    slots = manifest(size=args.items)
    if not (args.live or args.mock):
        count = sum(args.wave != "1" or s.wave == 1 for s in slots)
        print(f"Dry run: {len(endpoints)} × {count} slots; manifest SHA256 {fingerprint(slots)}; no network")
        return
    if args.mock:
        url = urlparse(args.mock)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1"} or
                not url.port or url.username or url.path not in {"", "/"} or url.query or url.fragment):
            p.error("--mock requires an explicit HTTP loopback host and port")
        host, key = args.mock.rstrip("/"), "mock"
    else:
        if scored and not args.confirm_scored:
            p.error("scored collection requires --live --confirm-scored")
        host, key = SERVER, os.environ.get("MODEL_EQUALITY_API_KEY")
        if not key:
            p.error("set MODEL_EQUALITY_API_KEY")
    count = collect(host + TASK_PATH, key, endpoints, slots, args.output, args.wave, scored)
    print(f"{count} saved answers: {args.output}")


if __name__ == "__main__":
    main()
