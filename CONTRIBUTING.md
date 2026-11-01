# Contributing to Conduit

Thanks for your interest in improving Conduit! This document explains how to
report issues, propose features, set up a development environment, and get your
changes merged.

By participating in this project you agree to abide by our
[Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug** — open an issue with steps to reproduce, expected vs. actual
  behaviour, your OS/Python version, and relevant logs.
- **Suggest a feature** — open an issue describing the problem, the proposed
  solution, and any alternatives you considered.
- **Improve docs** — fixes to the README, docstrings, or examples are very welcome.
- **Send code** — bugs fixes, tests, and focused features.

Please open an issue before starting significant work so we can agree on the
approach and avoid duplicated effort.

## Development setup

Conduit requires Python 3.10+.

```bash
git clone https://github.com/lochangowda10/Conduit.git
cd Conduit
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

Run the test suite (a single test may be flaky on Windows due to localhost
connection semantics; it passes on CI):

```bash
python -m pytest -q
```

Useful demo commands while developing:

```bash
conduit stats     # cache + routing demo
conduit tour      # walk through the concepts
conduit serve     # start the OpenAI-compatible gateway
```

## Project structure

```
conduit/            gateway package (imported as `conduit`)
  gateway.py        request pipeline, hedging, streaming
  cache.py          exact + semantic cache
  embeddings.py     hashing-trick embedder
  router.py         routing strategies and fallback chain
  resilience.py     retry, circuit breaker, logging decorators
  ratelimit.py      token bucket
  ledger.py         cost ledger
  telemetry.py      JSONL logs, CSV export
  analytics.py      pandas summaries
  server.py         OpenAI-compatible HTTP server (SSE streaming)
  providers/        provider ABC + simulated / OpenAI-compatible / local
benchmarks/         workload generator, benchmark runner, charts
demos/              concept tour and concurrency demo
tests/              pytest suite
```

## Coding guidelines

- Follow the existing style; keep changes small and focused.
- Add type hints to public functions and keep the typed error hierarchy intact
  (retry decisions are made by exception type, never by parsing messages).
- New behaviour must include tests. Prefer deterministic tests; use the
  simulated providers instead of hitting real APIs.
- Avoid adding heavy dependencies. The gateway core should stay lightweight.
- Do not log secrets or API keys. Never commit credentials or `config.json`.
- Update the README and docstrings when you change user-facing behaviour.

## Commit messages

Use clear, present-tense messages. [Conventional Commits](https://www.conventionalcommits.org/)
are encouraged, for example:

```
feat(router): add latency-aware tie-breaking
fix(cache): do not cache high-temperature requests
docs: clarify provider configuration
test: cover hedged request accounting
```

## Pull request process

1. Fork the repository and create a topic branch from `main`
   (e.g. `feat/semantic-cache`).
2. Make your changes and add or update tests.
3. Run `python -m pytest -q` and make sure everything passes.
4. Keep the PR focused; describe the motivation, the change, and how you tested it.
5. Link any related issues (e.g. `Closes #12`).
6. A maintainer will review the PR. Address review comments with new commits.

## Reporting security issues

Please do **not** open a public issue for security vulnerabilities. Report them
privately to the maintainer listed in `pyproject.toml` so a fix can be prepared.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE) that covers this project.
