"""End-to-end: a real HTTP server, called by Conduit's own OpenAI-compatible client.

Gateway A (simulated provider) is exposed over HTTP. Gateway B uses an
``OpenAICompatibleProvider`` pointed at A - so this exercises request
building, JSON parsing, SSE streaming and error mapping over a real socket.
"""

import json
import threading
import urllib.error
import urllib.request

import pytest

from conduit import ChatRequest, Gateway, GatewayConfig
from conduit.analytics import summarize, to_frame
from conduit.errors import ProviderUnavailable
from conduit.providers import OpenAICompatibleProvider
from conduit.server import make_server
from conduit.telemetry import RequestLogger, RequestRecord, TelemetryError, export_csv, load_jsonl


@pytest.fixture
def server(fast_provider):
    gw = Gateway([fast_provider("upstream")])
    srv = make_server(gw, port=0)  # port 0 = pick any free port
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv
    srv.shutdown()
    srv.server_close()
    gw.close()


def url(srv, path):
    return f"http://127.0.0.1:{srv.server_port}{path}"


def post(srv, body: bytes):
    req = urllib.request.Request(url(srv, "/v1/chat/completions"), data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=5)


def test_health_and_models(server):
    assert json.loads(urllib.request.urlopen(url(server, "/health")).read()) == {"status": "ok"}
    ids = [m["id"] for m in json.loads(urllib.request.urlopen(url(server, "/v1/models")).read())["data"]]
    assert "auto" in ids and "upstream" in ids


def test_completion_roundtrip(server):
    body = json.dumps({"messages": [{"role": "user", "content": "hello"}]}).encode()
    data = json.loads(post(server, body).read())
    assert data["object"] == "chat.completion" and data["x_conduit"]["provider"] == "upstream"


@pytest.mark.parametrize("body,code", [(b"{bad", 400), (b'{"messages": []}', 400), (b"[1]", 400)])
def test_bad_requests(server, body, code):
    with pytest.raises(urllib.error.HTTPError) as exc:
        post(server, body)
    assert exc.value.code == code


def test_gateway_chained_through_http(server):
    remote = OpenAICompatibleProvider("remote", model="auto", base_url=url(server, "/v1"), timeout_s=5)
    with Gateway([remote], GatewayConfig(cache_enabled=False)) as gw:
        r = gw.complete(ChatRequest.from_prompt("What is the capital of Peru?"))
        assert r.provider == "remote" and r.text.startswith("[sim-model]") and r.usage.prompt_tokens > 0
        streamed = "".join(gw.stream(ChatRequest.from_prompt("stream through http")))
        assert streamed.startswith("[sim-model]")


def test_connection_refused_maps_to_unavailable():
    p = OpenAICompatibleProvider("dead", model="m", base_url="http://127.0.0.1:9/v1", timeout_s=1)
    with pytest.raises(ProviderUnavailable):
        p.complete(ChatRequest.from_prompt("x"))


def test_auth(fast_provider):
    gw = Gateway([fast_provider()])
    srv = make_server(gw, port=0, api_key="secret")
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            post(srv, b'{"messages":[{"role":"user","content":"hi"}]}')
        assert exc.value.code == 401
    finally:
        srv.shutdown()
        srv.server_close()
        gw.close()


class TestTelemetry:
    def test_jsonl_roundtrip_with_corruption(self, tmp_path):
        log = RequestLogger(tmp_path / "logs" / "r.jsonl")
        for i in range(3):
            log.log(RequestRecord(f"r{i}", latency_ms=float(i)))
        with (tmp_path / "logs" / "r.jsonl").open("a") as fh:
            fh.write("{oops\n\n")
        records, stats = load_jsonl(tmp_path / "logs" / "r.jsonl")
        assert len(records) == 3 and stats == {"ok": 3, "blank": 1, "malformed": 1}

    def test_missing_file(self, tmp_path):
        with pytest.raises(TelemetryError):
            load_jsonl(tmp_path / "nope.jsonl")

    def test_csv_and_analytics(self, tmp_path):
        recs = [RequestRecord("a", scenario="s", latency_ms=100, cost_usd=0.01, cache="miss"),
                RequestRecord("b", scenario="s", latency_ms=1, cost_usd=0.0, cache="exact"),
                RequestRecord("c", scenario="s", status="error")]
        assert export_csv(recs, tmp_path / "x.csv") == 3
        summary = summarize(to_frame(recs))
        row = summary.loc["s"]
        assert row["requests"] == 3 and row["success_rate"] == pytest.approx(2 / 3)
        assert row["cache_hit_rate"] == pytest.approx(1 / 3) and row["total_cost_usd"] == pytest.approx(0.01)
