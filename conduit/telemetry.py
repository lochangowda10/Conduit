"""Request logging (JSON Lines) with CSV export and defensive loading.

Syllabus link
-------------
* Experiment 7 (CSV + JSON + exception handling): logs are written as JSON
  Lines, exported to CSV with ``csv.DictWriter``, and loaded back with
  explicit handling of missing files, permission problems, blank lines and
  malformed/corrupted lines.
* Experiment 8 (Multithreading): many request threads log concurrently, so
  writes go through a lock (otherwise two lines can interleave mid-write).
* Experiment 6 (Generators): ``iter_jsonl`` streams a huge log file one
  record at a time instead of loading it all into memory.

Why JSON Lines and not one big JSON array?
------------------------------------------
* Append-only: each request appends one line; no need to rewrite the file.
* Crash-safe: a crash corrupts at most the last line, not the whole file.
* Streamable: tools (and our generator) can process it line by line.
"""

from __future__ import annotations

import csv
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterator


@dataclass
class RequestRecord:
    request_id: str
    ts: float = field(default_factory=time.time)
    scenario: str = ""
    status: str = "ok"            # "ok" | "error"
    provider: str | None = None
    model: str | None = None
    cache: str = "miss"           # "miss" | "exact" | "semantic" | "bypass"
    similarity: float | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    attempts: int = 0
    hedged: bool = False
    tier: str | None = None
    complexity: float | None = None
    error_type: str | None = None
    # benchmark-only ground truth (empty in production)
    label: str | None = None
    cached_label: str | None = None  # label of the cache entry that answered (to detect false hits)

    @classmethod
    def columns(cls) -> list[str]:
        return [f.name for f in fields(cls)]


class TelemetryError(Exception):
    """Raised when a log file cannot be read at all."""


class RequestLogger:
    """Collects records in memory and optionally appends them to a JSONL file."""

    def __init__(self, path: str | os.PathLike | None = None, keep_in_memory: bool = True):
        self.path = Path(path) if path else None
        self.keep_in_memory = keep_in_memory
        self.records: list[RequestRecord] = []
        self._lock = threading.Lock()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: RequestRecord) -> None:
        line = json.dumps(asdict(record), separators=(",", ":"))
        with self._lock:
            if self.keep_in_memory:
                self.records.append(record)
            if self.path:
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[RequestRecord]:
        with self._lock:
            snapshot = list(self.records)
        return iter(snapshot)


# ---------------------------------------------------------------------------
# Loading & exporting
# ---------------------------------------------------------------------------
def iter_jsonl(path: str | os.PathLike, stats: dict[str, int] | None = None) -> Iterator[dict[str, Any]]:
    """Yield one dict per valid line. Bad lines are counted, not fatal.

    Raises ``TelemetryError`` only if the file itself can't be opened.
    """
    stats = stats if stats is not None else {}
    stats.setdefault("ok", 0)
    stats.setdefault("blank", 0)
    stats.setdefault("malformed", 0)
    p = Path(path)
    try:
        fh = p.open("r", encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise TelemetryError(f"log file not found: {p}") from exc
    except PermissionError as exc:
        raise TelemetryError(f"no permission to read: {p}") from exc
    except IsADirectoryError as exc:
        raise TelemetryError(f"expected a file but got a directory: {p}") from exc
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                stats["blank"] += 1
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError("line is not a JSON object")
            except (json.JSONDecodeError, ValueError):
                stats["malformed"] += 1
                continue
            stats["ok"] += 1
            yield obj


def load_jsonl(path: str | os.PathLike) -> tuple[list[dict[str, Any]], dict[str, int]]:
    stats: dict[str, int] = {}
    records = list(iter_jsonl(path, stats))
    return records, stats


def export_csv(records: list[RequestRecord] | list[dict[str, Any]], path: str | os.PathLike) -> int:
    """Write records to CSV. Unknown keys are ignored, missing keys left blank."""
    rows = [asdict(r) if isinstance(r, RequestRecord) else r for r in records]
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RequestRecord.columns(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def export_jsonl(records: list[RequestRecord], path: str | os.PathLike) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(asdict(r), separators=(",", ":")) + "\n")
    return len(records)
