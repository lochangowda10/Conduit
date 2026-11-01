"""Command-line interface:  python -m conduit <command> [options]

  serve    start the OpenAI-compatible HTTP gateway
  chat     send one prompt through the gateway and print the answer + routing info
  stats    run a few prompts and print the gateway's metrics (quick demo)
  bench    run the 5-scenario benchmark       (same as: python -m benchmarks.run_benchmark)
  report   draw the charts from benchmark results (same as: python -m benchmarks.report)
  tour     walk through every syllabus concept, with live output
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from .config import GatewayConfig
from .errors import ConduitError
from .gateway import Gateway
from .types import ChatRequest

DEFAULT_PROVIDERS = [
    {"type": "simulated", "name": "sim-small", "tier": "small", "model": "sim-mini",
     "input_price_per_m": 0.10, "output_price_per_m": 0.40, "latency_median_ms": 350},
    {"type": "simulated", "name": "sim-medium", "tier": "medium", "model": "sim-mid",
     "input_price_per_m": 0.50, "output_price_per_m": 2.00, "latency_median_ms": 650},
    {"type": "simulated", "name": "sim-large", "tier": "large", "model": "sim-frontier",
     "input_price_per_m": 3.00, "output_price_per_m": 15.00, "latency_median_ms": 1100},
]


def load_gateway(config_path: str | None) -> Gateway:
    if config_path:
        return Gateway.from_json(config_path)
    return Gateway.from_config(GatewayConfig(providers=DEFAULT_PROVIDERS))


def _run_module(module: str, argv: list[str]) -> int:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))  # so `benchmarks` is importable from anywhere
    import importlib

    mod = importlib.import_module(module)
    sys.argv = [module] + argv
    mod.main()
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="conduit", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="start the HTTP gateway")
    p_serve.add_argument("--config", help="JSON config file (default: built-in simulated providers)")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--api-key", default=os.environ.get("CONDUIT_API_KEY"),
                         help="require this bearer token from clients")

    p_chat = sub.add_parser("chat", help="send one prompt")
    p_chat.add_argument("prompt")
    p_chat.add_argument("--config")
    p_chat.add_argument("--model", default="auto", help="auto | small | medium | large | <provider name>")
    p_chat.add_argument("--stream", action="store_true")

    p_stats = sub.add_parser("stats", help="send a few prompts and print metrics")
    p_stats.add_argument("--config")

    sub.add_parser("bench", help="run the benchmark", add_help=False)
    sub.add_parser("report", help="draw charts", add_help=False)
    sub.add_parser("tour", help="syllabus concept tour", add_help=False)

    args, rest = parser.parse_known_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")

    try:
        if args.cmd == "bench":
            return _run_module("benchmarks.run_benchmark", rest)
        if args.cmd == "report":
            return _run_module("benchmarks.report", rest)
        if args.cmd == "tour":
            return _run_module("demos.concepts_tour", rest)
        if args.cmd == "serve":
            from .server import serve

            serve(load_gateway(args.config), args.host, args.port, args.api_key)
            return 0
        if args.cmd == "chat":
            with load_gateway(args.config) as gw:
                req = ChatRequest.from_prompt(args.prompt, model=args.model)
                if args.stream:
                    for chunk in gw.stream(req):
                        print(chunk, end="", flush=True)
                    print()
                    print(json.dumps(gw.stats()["ledger"]["total"]))
                else:
                    resp = gw.complete(req)
                    print(resp.text)
                    print(json.dumps(resp.to_openai()["x_conduit"], indent=2))
            return 0
        if args.cmd == "stats":
            prompts = [
                "What is the capital of France?",
                "What is the capital of France?",          # exact repeat -> exact cache hit
                "Tell me the capital of France, please.",  # paraphrase -> semantic cache hit
                "What is the capital of Spain?",           # near-miss -> must NOT hit
                "Explain recursion in simple terms.",
                "Design a distributed rate limiter. Compare the trade-offs of two approaches, "
                "analyze failure modes step by step and explain how it scales.",
            ]
            with load_gateway(args.config) as gw:
                for p in prompts:
                    r = gw.complete(ChatRequest.from_prompt(p))
                    print(f"{r.provider:<12} cache={r.cache:<9} {r.latency_ms:7.1f} ms  {str(r.usage):<26} {p[:50]}")
                print(json.dumps(gw.stats(), indent=2))
            return 0
    except ConduitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
