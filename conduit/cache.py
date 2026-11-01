"""Two-level response cache: exact match + semantic (embedding) match.

Syllabus link
-------------
* Experiment 9 (NumPy): the semantic lookup is a single vectorised
  matrix-vector product over a pre-allocated (capacity x dim) matrix, plus
  boolean masking and ``argmax``. No Python loop over entries.
* Experiment 8 (Multithreading): an ``RLock`` guards all mutable state since
  every request thread reads and writes the cache.
* Experiment 1 (OOP): ``CacheStats`` exposes a derived ``hit_rate`` property.

Design decisions (good interview material)
------------------------------------------
1. Exact cache first: a dict lookup on a SHA-256 key is O(1) and has zero
   false-positive risk, so it is always tried before the fuzzy match.
2. Semantic cache second: paraphrases ("What's France's capital?" vs
   "What is the capital of France?") hit even though the strings differ.
3. Scope: two prompts only match if they share the same *scope* (system
   prompt + model group + temperature bucket). The same question under a
   different system prompt may deserve a different answer.
4. Only cache deterministic-ish requests (temperature <= ``max_temperature``).
   Caching a temperature=1.0 creative request would kill the variety the
   caller asked for.
5. Entity guard: embeddings are bad at telling "capital of France" from
   "capital of Spain" (they share most words). We refuse a semantic hit if
   the two prompts contain different numbers or proper nouns.
6. IDF weighting: words that appear in almost every prompt ("explain",
   "python", "what is") carry little meaning, rare words carry a lot. We
   keep a document-frequency count per hash bucket and re-weight vectors
   *at query time*, so old entries benefit from the latest statistics.
7. Bounded memory: LRU eviction plus TTL expiry.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .embeddings import Embedder, HashingEmbedder
from .types import ChatRequest, ChatResponse

_QUOTED_RE = re.compile(r"\"([^\"]+)\"|“([^”]+)”|(?:^|(?<=\s))'([^']+)'")
_WORD_RE = re.compile(r"\d+(?:\.\d+)?|[A-Za-z][A-Za-z']*|[.!?:]")
# Capitalised only because they start a sentence, so not salient.
_COMMON_STARTERS = frozenset(
    """explain write give tell translate describe list summarize summarise compare define
    convert calculate find create generate what how why when where who which is are can
    could would should do does please i hey hi hello briefly quickly quick question thanks
    thank keep be need help ok okay so also the a an in on in my your this that show
    walk debug design summarize analyze give suggest tell""".split()
)


def salient_tokens(text: str) -> frozenset[str]:
    """Numbers, quoted strings and proper nouns.

    These are the tokens where 'almost the same' is *not* good enough:
    "What is 12 * 7?" must never be answered with the cached result of
    "What is 12 * 8?", and "capital of Spain" must not reuse "capital of France".
    """
    found: set[str] = set()
    for groups in _QUOTED_RE.findall(text):
        found.update(g.strip().lower() for g in groups if g)
    for match in _WORD_RE.finditer(text):
        tok = match.group()
        if tok in ".!?:":
            continue
        if tok[0].isdigit():
            found.add(tok)
        elif tok[0].isupper():
            word = tok.lower().split("'")[0]  # "France's" -> "france", "I'm" -> "i"
            if word not in _COMMON_STARTERS:
                found.add(word)
    return frozenset(found)


@dataclass
class CacheStats:
    lookups: int = 0
    exact_hits: int = 0
    semantic_hits: int = 0
    guard_rejections: int = 0
    evictions: int = 0
    expirations: int = 0

    @property
    def hits(self) -> int:
        return self.exact_hits + self.semantic_hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict:
        return {**self.__dict__, "hits": self.hits, "hit_rate": round(self.hit_rate, 4)}


@dataclass
class CacheEntry:
    key: str
    scope: str
    prompt: str
    response_text: str
    provider: str
    model: str
    salient: frozenset[str]
    created: float
    meta: dict


class ResponseCache:
    def __init__(
        self,
        embedder: Embedder | None = None,
        capacity: int = 10_000,
        similarity_threshold: float = 0.85,
        ttl_seconds: float = 3600.0,
        max_temperature: float = 0.3,
        semantic_enabled: bool = True,
        entity_guard: bool = True,
        idf_weighting: bool = True,
        clock: Callable[[], float] = time.time,
    ):
        if not 0.0 < similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold must be in (0, 1]")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.embedder = embedder or HashingEmbedder()
        self.capacity = capacity
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.max_temperature = max_temperature
        self.semantic_enabled = semantic_enabled
        self.entity_guard = entity_guard
        self.idf_weighting = idf_weighting
        self._clock = clock
        self._lock = threading.RLock()

        # Exact layer: key -> slot index, ordered by recency (LRU).
        self._exact: OrderedDict[str, int] = OrderedDict()
        # Semantic layer: pre-allocated arrays indexed by slot.
        self._vectors = np.zeros((capacity, self.embedder.dim), dtype=np.float32)
        self._valid = np.zeros(capacity, dtype=bool)
        self._scope_ids = np.full(capacity, -1, dtype=np.int64)
        self._entries: list[CacheEntry | None] = [None] * capacity
        self._free_slots = list(range(capacity - 1, -1, -1))
        self._scope_index: dict[str, int] = {}
        # Document frequency per hash bucket, for IDF weighting (see idf_weights()).
        self._df = np.zeros(self.embedder.dim, dtype=np.float64)
        self._n_docs = 0
        self.stats = CacheStats()

    # ------------------------------------------------------------------ keys
    @staticmethod
    def scope_of(request: ChatRequest) -> str:
        temp_bucket = round(request.temperature, 1)
        system_hash = hashlib.sha256(request.system_text.encode()).hexdigest()[:16]
        return f"{request.model}|{temp_bucket}|{system_hash}|{request.max_tokens}"

    @staticmethod
    def exact_key(request: ChatRequest) -> str:
        payload = json.dumps(
            {
                "scope": ResponseCache.scope_of(request),
                "messages": [m.to_dict() for m in request.messages],
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def cacheable(self, request: ChatRequest) -> bool:
        return request.temperature <= self.max_temperature

    def _scope_id(self, scope: str) -> int:
        if scope not in self._scope_index:
            self._scope_index[scope] = len(self._scope_index)
        return self._scope_index[scope]

    # ---------------------------------------------------------------- lookup
    def get(self, request: ChatRequest) -> tuple[CacheEntry, str, float] | None:
        """Return ``(entry, kind, similarity)`` or ``None``.

        ``kind`` is "exact" or "semantic".
        """
        if not self.cacheable(request):
            return None
        key = self.exact_key(request)
        scope = self.scope_of(request)
        with self._lock:
            self.stats.lookups += 1
            now = self._clock()

            slot = self._exact.get(key)
            if slot is not None:
                entry = self._entries[slot]
                if entry is not None and now - entry.created <= self.ttl_seconds:
                    self._exact.move_to_end(key)  # mark as recently used
                    self.stats.exact_hits += 1
                    return entry, "exact", 1.0
                self._evict_slot(slot, expired=True)

            if not self.semantic_enabled or scope not in self._scope_index or not self._valid.any():
                return None
            query_vec = self.embedder.embed(request.prompt_text)
            return self._semantic_lookup(query_vec, self._scope_index[scope], request.prompt_text, now)

    def idf_weights(self) -> np.ndarray:
        """Smoothed inverse document frequency for every hash bucket.

        idf = ln((1 + N) / (1 + df)) + 1, the same formula scikit-learn uses.
        Buckets that appear in almost every prompt ("explain", "python",
        "what is") get weight ~1; rare ones ("backpropagation", "kenya")
        get a larger weight, so they dominate the similarity.
        """
        return (np.log((1.0 + self._n_docs) / (1.0 + self._df)) + 1.0).astype(np.float32)

    def similarities(self, query_vec: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """Cosine similarity between the query and the given cache rows (vectorised)."""
        matrix = self._vectors[rows]                              # (k, dim) fancy indexing
        if self.idf_weighting:
            w = self.idf_weights()
            matrix = matrix * w                                   # broadcasting (k, dim) * (dim,)
            query_vec = query_vec * w
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        return (matrix @ query_vec) / np.clip(norms, 1e-12, None)

    def _semantic_lookup(self, query_vec: np.ndarray, scope_id: int, prompt: str, now: float):
        eligible = np.flatnonzero(self._valid & (self._scope_ids == scope_id))  # boolean mask -> indices
        if eligible.size == 0:
            return None
        # One vectorised pass: similarities for ALL eligible entries at once.
        sims = self.similarities(query_vec, eligible)
        # Check the best few candidates, so a guard rejection on #1 can fall to #2.
        k = min(5, eligible.size)
        order = np.argpartition(-sims, k - 1)[:k]
        order = order[np.argsort(-sims[order])]
        query_salient = salient_tokens(prompt) if self.entity_guard else frozenset()
        for pos in order:
            slot = int(eligible[pos])
            sim = float(sims[pos])
            if sim < self.similarity_threshold:
                break
            entry = self._entries[slot]
            if entry is None:
                continue
            if now - entry.created > self.ttl_seconds:
                self._evict_slot(slot, expired=True)
                continue
            if self.entity_guard and entry.salient != query_salient:
                self.stats.guard_rejections += 1
                continue
            self._exact.move_to_end(entry.key)
            self.stats.semantic_hits += 1
            return entry, "semantic", sim
        return None

    # ----------------------------------------------------------------- store
    def put(self, request: ChatRequest, response: ChatResponse, meta: dict | None = None) -> None:
        if not self.cacheable(request):
            return
        key = self.exact_key(request)
        scope = self.scope_of(request)
        vec = self.embedder.embed(request.prompt_text) if self.semantic_enabled else None
        with self._lock:
            if key in self._exact:  # refresh existing entry
                self._evict_slot(self._exact[key])
            if not self._free_slots:
                lru_key = next(iter(self._exact))  # OrderedDict: first = least recently used
                self._evict_slot(self._exact[lru_key])
                self.stats.evictions += 1
            slot = self._free_slots.pop()
            self._entries[slot] = CacheEntry(
                key=key,
                scope=scope,
                prompt=request.prompt_text,
                response_text=response.text,
                provider=response.provider,
                model=response.model,
                salient=salient_tokens(request.prompt_text),
                created=self._clock(),
                meta=meta or {},
            )
            self._exact[key] = slot
            if vec is not None:
                self._vectors[slot] = vec
                self._valid[slot] = True
                self._scope_ids[slot] = self._scope_id(scope)
                self._df += vec != 0  # bool array adds as 0/1: one more "document" per bucket used
                self._n_docs += 1

    def _evict_slot(self, slot: int, expired: bool = False) -> None:
        entry = self._entries[slot]
        if entry is None:
            return
        self._exact.pop(entry.key, None)
        self._entries[slot] = None
        self._valid[slot] = False
        self._scope_ids[slot] = -1
        self._free_slots.append(slot)
        if expired:
            self.stats.expirations += 1

    def clear(self) -> None:
        with self._lock:
            for slot in list(self._exact.values()):
                self._evict_slot(slot)

    def __len__(self) -> int:
        with self._lock:
            return len(self._exact)

    def __contains__(self, request: ChatRequest) -> bool:
        with self._lock:
            return self.exact_key(request) in self._exact

    def __repr__(self) -> str:
        return (
            f"ResponseCache(size={len(self)}/{self.capacity}, "
            f"threshold={self.similarity_threshold}, hit_rate={self.stats.hit_rate:.1%})"
        )
