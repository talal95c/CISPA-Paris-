"""Expose one local HF family at the reference OpenAI-compatible routes."""
import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

import torch

from collect import TASK_PATH
from protocol import Slot
from simulate import LocalModel, VARIANTS


def make_handler(local):
    variants = {f"ref-{i+1:02d}": v for i, v in enumerate(VARIANTS)}
    lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, data):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != f"{TASK_PATH}/models":
                self.send_json(404, {"detail": "unknown route"})
                return
            self.send_json(200, {"data": [{"id": endpoint} for endpoint in variants]})

        def do_POST(self):
            if self.path != f"{TASK_PATH}/chat/completions":
                self.send_json(404, {"detail": "unknown route"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 1 <= size <= 16384:
                    raise ValueError("invalid body size")
                body = json.loads(self.rfile.read(size))
                variant = variants[body["model"]]
                count = body.get("max_tokens", 256)
                messages = body["messages"]
                if (type(count) is not int or not 1 <= count <= 256 or body.get("n", 1) != 1 or
                        len(messages) != 1 or messages[0]["role"] != "user" or
                        not isinstance(messages[0]["content"], str)):
                    raise ValueError("expected one user message, n=1 and max_tokens in [1, 256]")
                slot = Slot("http", "long", 1, messages[0]["content"])
                with lock:
                    originals = local.configure(variant)
                    try:
                        text, _ = next(local.generate([slot], variant, count))
                    finally:
                        local.restore(originals)
            except (KeyError, TypeError, IndexError, ValueError, AttributeError) as exc:
                self.send_json(400, {"detail": str(exc)})
                return
            self.send_json(200, {"choices": [{"message": {"content": text}}]})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", type=int, choices=(0, 1), default=0)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    local = LocalModel(args.family, args.offline)
    with ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(local)) as server:
        print(f"http://127.0.0.1:{server.server_port}", flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
