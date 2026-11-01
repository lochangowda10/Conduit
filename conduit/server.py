"""OpenAI-compatible HTTP server (standard library only).

Any app that already uses the OpenAI SDK can switch to Conduit by changing
one line - the base URL:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="anything")
    client.chat.completions.create(model="auto", messages=[...])

Endpoints
---------
POST /v1/chat/completions   normal JSON response, or SSE stream if "stream": true
GET  /v1/models             logical models (auto/small/medium/large) + providers
GET  /health                liveness probe
GET  /metrics               cost ledger, cache stats, breaker states, provider latency

Syllabus link
-------------
* Experiment 8 (Multithreading): ``ThreadingHTTPServer`` handles each
  connection in its own thread. LLM calls are I/O-bound (the thread waits on
  the network), and blocking I/O releases the GIL, so threads give real
  concurrency here despite the GIL.
* Experiment 6 (Generators): streaming responses forward the gateway's
  generator to the socket chunk by chunk (Server-Sent Events).
* Experiment 7 (JSON + exception handling): malformed bodies map to 400,
  oversized bodies to 413, gateway errors to their own status codes.

Why not FastAPI? FastAPI (async) would be the choice for a production
service with thousands of concurrent streams. The standard-library version
keeps the project zero-dependency (it runs on any teammate's laptop) and
makes the threading model explicit, which is the point of this course.
The gateway core is framework-agnostic, so swapping the HTTP layer later
is a small change.
"""

from __future__ import annotations

import json
import logging
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from .errors import ConduitError
from .gateway import Gateway
from .types import ChatRequest

log = logging.getLogger("conduit.server")

MAX_BODY_BYTES = 1_000_000


class ConduitHTTPServer(ThreadingHTTPServer):
    daemon_threads = True  # don't block shutdown on open connections

    def __init__(self, address: tuple[str, int], gateway: Gateway, api_key: str | None = None):
        super().__init__(address, ConduitHandler)
        self.gateway = gateway
        self.api_key = api_key


class ConduitHandler(BaseHTTPRequestHandler):
    server: ConduitHTTPServer
    protocol_version = "HTTP/1.1"

    # ----- helpers -------------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # route access logs to logging
        log.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, payload: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str, err_type: str) -> None:
        self._send_json(status, {"error": {"message": message, "type": err_type}})

    def _authorized(self) -> bool:
        if not self.server.api_key:
            return True
        return self.headers.get("Authorization", "") == f"Bearer {self.server.api_key}"

    def _read_json(self) -> dict[str, Any] | None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._send_error(400, "invalid Content-Length", "invalid_request_error")
            return None
        if length > MAX_BODY_BYTES:
            self._send_error(413, "request body too large", "invalid_request_error")
            return None
        raw = self.rfile.read(length) if length else b""
        try:
            data = json.loads(raw or b"{}")
        except json.JSONDecodeError as exc:
            self._send_error(400, f"invalid JSON: {exc.msg}", "invalid_request_error")
            return None
        if not isinstance(data, dict):
            self._send_error(400, "JSON body must be an object", "invalid_request_error")
            return None
        return data

    # ----- routes ----------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802 - name required by BaseHTTPRequestHandler
        gw = self.server.gateway
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/metrics":
            self._send_json(200, gw.stats())
        elif self.path == "/v1/models":
            logical = [{"id": m, "object": "model", "owned_by": "conduit"} for m in ("auto", "small", "medium", "large")]
            physical = [{"id": p.name, "object": "model", "owned_by": p.kind} for p in gw.providers]
            self._send_json(200, {"object": "list", "data": logical + physical})
        else:
            self._send_error(404, f"no route for GET {self.path}", "not_found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._send_error(404, f"no route for POST {self.path}", "not_found")
            return
        if not self._authorized():
            self._send_error(401, "invalid or missing API key", "authentication_error")
            return
        payload = self._read_json()
        if payload is None:
            return
        try:
            request = ChatRequest.from_openai(payload)
        except (ValueError, TypeError) as exc:
            self._send_error(400, str(exc), "invalid_request_error")
            return
        if request.stream:
            self._stream(request)
        else:
            self._complete(request)

    def _complete(self, request: ChatRequest) -> None:
        try:
            response = self.server.gateway.complete(request)
        except ConduitError as exc:
            self._send_error(getattr(exc, "status_code", 500), str(exc), type(exc).__name__)
            return
        headers = {"X-Conduit-Provider": response.provider, "X-Conduit-Cache": response.cache}
        self._send_json(200, response.to_openai(), headers)

    def _stream(self, request: ChatRequest) -> None:
        gen = self.server.gateway.stream(request)
        try:
            first = next(gen)  # surface routing/provider errors as a normal HTTP error
        except StopIteration:
            first = ""
        except ConduitError as exc:
            self._send_error(getattr(exc, "status_code", 500), str(exc), type(exc).__name__)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        created = int(time.time())

        def event(delta: dict[str, Any], finish: str | None = None) -> bytes:
            chunk = {
                "id": f"chatcmpl-{request.request_id}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": request.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            return f"data: {json.dumps(chunk)}\n\n".encode()

        try:
            self.wfile.write(event({"role": "assistant", "content": first}))
            self.wfile.flush()
            for piece in gen:
                self.wfile.write(event({"content": piece}))
                self.wfile.flush()
            self.wfile.write(event({}, finish="stop"))
            self.wfile.write(b"data: [DONE]\n\n")
        except ConduitError as exc:
            # Headers are already sent, so we can't change the status code:
            # tell the client in-band, then close.
            err = {"error": {"message": str(exc), "type": type(exc).__name__}}
            self.wfile.write(f"data: {json.dumps(err)}\n\n".encode())
        except (BrokenPipeError, ConnectionResetError):
            log.info("client disconnected mid-stream")
            gen.close()  # stop generating tokens nobody will read


def make_server(gateway: Gateway, host: str = "127.0.0.1", port: int = 8080, api_key: str | None = None) -> ConduitHTTPServer:
    return ConduitHTTPServer((host, port), gateway, api_key)


def serve(gateway: Gateway, host: str = "127.0.0.1", port: int = 8080, api_key: str | None = None) -> None:
    server = make_server(gateway, host, port, api_key)
    print(f"Conduit listening on http://{host}:{server.server_port}/v1  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
        gateway.close()
