#!/usr/bin/env python3
"""Minimal localhost proxy that adds the Qwen EOS stop sequence.

MLX-LM's OpenAI-compatible server otherwise returns the literal
``<|im_end|>`` token in non-streaming response text.  This proxy changes no
prompt or sampling field; it only adds that tokenizer-declared EOS string to
the request's stop list and forwards the response unchanged.  It is intended
only for the frozen local-control experiment.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import urllib.error
import urllib.request


class Handler(BaseHTTPRequestHandler):
    upstream = "http://127.0.0.1:8080"

    def _forward(self, body: bytes | None = None) -> None:
        headers = {"Content-Type": self.headers.get("Content-Type", "application/json")}
        request = urllib.request.Request(
            self.upstream + self.path,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(request, timeout=1800) as response:
                payload = response.read()
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get(
                    "Content-Type", "application/json"))
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
        except urllib.error.HTTPError as exc:
            payload = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        size = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(size) or b"{}")
        stops = payload.get("stop")
        if stops is None:
            payload["stop"] = ["<|im_end|>"]
        elif isinstance(stops, str):
            payload["stop"] = [stops, "<|im_end|>"] if stops != "<|im_end|>" else [stops]
        elif "<|im_end|>" not in stops:
            payload["stop"] = [*stops, "<|im_end|>"]
        self._forward(json.dumps(payload).encode())

    def log_message(self, fmt: str, *args) -> None:
        print("local-stop-proxy:", fmt % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--upstream", default="http://127.0.0.1:8080")
    args = parser.parse_args()
    Handler.upstream = args.upstream.rstrip("/")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"local stop proxy at http://{args.host}:{args.port} -> {Handler.upstream}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
