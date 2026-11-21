import json
import tempfile
import threading
import unittest
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from collect import TASK_PATH, collect, main
from protocol import manifest


class CollectionTests(unittest.TestCase):
    def test_dry_run_does_not_call_http(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            with patch("sys.argv", ["collect.py", "--target", "refs", "--dry-run", "--manifest-out", str(path)]), \
                    patch("collect.request") as send:
                main()
            self.assertEqual(json.loads(path.read_text())[34]["id"], "burst-01")
        send.assert_not_called()

    def test_manifest(self):
        slots = manifest()
        self.assertEqual(Counter(s.kind for s in slots), {"flag": 4, "long": 30, "burst": 66})
        self.assertEqual(sum(s.wave == 1 for s in slots), 14)
        self.assertEqual(len({s.id for s in slots}), 100)
        self.assertEqual(slots[34].prompt, slots[94].prompt)
        self.assertEqual(slots[35].prompt, slots[95].prompt)
        self.assertTrue(all(len(s.items) == len({i.tag for i in s.items}) == 20 for s in slots[34:]))

    def test_resume_budget_reserve_and_retry_on_loopback(self):
        used = Counter()
        busy = set()
        endpoints = ["ep-01", "ep-02", "ref-01"]
        slots = manifest()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def send_json(self, status, body, retry=False):
                self.send_response(status)
                if retry:
                    self.send_header("Retry-After", "0")
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_GET(self):
                assert self.path == TASK_PATH + "/models"
                self.send_json(200, {"data": [{"id": ep, "budget": {"used": used[ep], "limit": 100}} for ep in endpoints]})

            def do_POST(self):
                assert self.path == TASK_PATH + "/chat/completions"
                assert self.headers["X-API-Key"] == "test"
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                assert payload["max_tokens"] == 256
                ep = payload["model"]
                if ep not in busy:
                    busy.add(ep)
                    self.send_json(429, {"detail": "busy"}, retry=True)
                    return
                used[ep] += 1
                content = "" if ep == "ep-01" and used[ep] in {1, 2, 97} else "test"
                self.send_json(200, {"choices": [{"message": {"content": content}}], "extra": {"preserved": True},
                                     "budget": {"used": used[ep], "limit": 100}})

        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server, tempfile.TemporaryDirectory() as tmp:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}{TASK_PATH}"
                journal = Path(tmp) / "scored.jsonl"
                self.assertEqual(collect(base, "test", endpoints[:2], slots, journal, "1", True), 28)
                self.assertEqual(collect(base, "test", endpoints[:2], slots, journal, "all", True), 200)
                last_size = journal.stat().st_size
                self.assertEqual(collect(base, "test", endpoints[:2], slots, journal, "all", True), 200)
                self.assertEqual(journal.stat().st_size, last_size)
                self.assertEqual(collect(base, "test", endpoints[2:], slots, Path(tmp) / "refs.jsonl", "1", False), 14)
                rows = [json.loads(line) for line in journal.read_text().splitlines()]
                self.assertEqual(used, {"ep-01": 100, "ep-02": 100, "ref-01": 14})
                self.assertEqual(sum(row["event"] == "busy" for row in rows), 2)
                repair = next(row for row in rows if row["event"] == "answer" and row["endpoint"] == "ep-01"
                              and row["id"] == "reserve-01")
                self.assertEqual(repair["probe_id"], "flag-01")
                self.assertEqual(repair["response"], "")
                self.assertEqual(repair["body"]["extra"], {"preserved": True})
                attempts = {row["id"]: row["probe_id"] for row in rows if row["event"] == "answer"
                            and row["endpoint"] == "ep-01" and row["id"].startswith("reserve-")}
                self.assertEqual(attempts, {"reserve-01": "flag-01", "reserve-02": "flag-02",
                                            "reserve-03": "reserve-03", "reserve-04": "reserve-04"})
            finally:
                server.shutdown()
                thread.join()


if __name__ == "__main__":
    unittest.main()
