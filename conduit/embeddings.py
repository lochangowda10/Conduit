"""Text embeddings for the semantic cache.

Syllabus link
-------------
* Experiment 9 (NumPy): vectors are built with ``np.add.at`` (scatter-add),
  normalised with ``np.linalg.norm`` along an axis, and compared with a single
  matrix-vector product (see ``cache.py``).
* Experiment 4 (Abstract classes): ``Embedder`` is an ABC; the hashing and
  sentence-transformer embedders are interchangeable implementations.

Why a hashing embedder?
-----------------------
A neural embedding model (e.g. sentence-transformers) is more accurate but
needs a ~100 MB download and ideally a GPU. The *hashing trick* (Weinberger
et al., 2009) maps words/n-grams straight to vector positions with a hash
function, so it needs no training, no vocabulary and no internet, and it
runs on any laptop. The trade-off - lower paraphrase recall - is measured in
``benchmarks/cache_eval.py`` rather than hidden.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from functools import lru_cache

import numpy as np

_CONTRACTIONS = {
    "what's": "what is", "it's": "it is", "how's": "how is", "who's": "who is",
    "where's": "where is", "that's": "that is", "i'm": "i am", "can't": "cannot",
    "don't": "do not", "doesn't": "does not", "isn't": "is not", "won't": "will not",
}

_STOPWORDS = frozenset(
    """a an the is are was were be been being of to in on for with by at from as and or
    but if then so than that this these those it its me my i you your we our they their
    please can could would should will do does did tell give show what which who whom how
    about into over under again just very also any some""".split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    for short, full in _CONTRACTIONS.items():
        text = text.replace(short, full)
    text = re.sub(r"'s\b", "", text)  # "france's" -> "france"
    return text


def content_tokens(text: str) -> list[str]:
    """Lower-cased tokens with stopwords removed."""
    return [t for t in _TOKEN_RE.findall(normalize(text)) if t not in _STOPWORDS]


@lru_cache(maxsize=65536)
def _stable_hash(feature: str) -> int:
    """Deterministic 64-bit hash.

    Python's built-in ``hash()`` is randomised per process (PYTHONHASHSEED),
    so vectors computed in one run would not match another run's. A
    cryptographic digest is stable everywhere.
    """
    return int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")


class Embedder(ABC):
    """Interface every embedder must satisfy."""

    dim: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D float32 vector of length ``dim`` with L2 norm 1."""

    def embed_many(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) matrix. Subclasses may vectorise this."""
        return np.vstack([self.embed(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class HashingEmbedder(Embedder):
    """Signed feature hashing over word unigrams, word bigrams and char trigrams.

    * unigrams carry the meaning ("capital", "france")
    * bigrams carry a little word order ("linked list" vs "list linked")
    * char trigrams make it robust to typos and inflections ("explain"/"explaining")

    Each feature hashes to an index; a second hash bit picks the sign (+/-),
    which makes collisions cancel out on average instead of piling up.
    """

    def __init__(self, dim: int = 1024, unigram_w: float = 1.0, bigram_w: float = 0.5, char_w: float = 0.25):
        if dim < 16 or dim & (dim - 1):
            raise ValueError("dim must be a power of two >= 16")
        self.dim = dim
        self.weights = {"u": unigram_w, "b": bigram_w, "c": char_w}

    def _features(self, text: str) -> list[tuple[str, float]]:
        toks = content_tokens(text)
        feats: list[tuple[str, float]] = [(f"u:{t}", self.weights["u"]) for t in toks]
        feats += [(f"b:{a}_{b}", self.weights["b"]) for a, b in zip(toks, toks[1:])]
        for t in toks:
            padded = f"#{t}#"
            n_grams = max(1, len(padded) - 2)
            # Spread a word's char-gram weight so long words don't dominate.
            w = self.weights["c"] / n_grams
            feats += [(f"c:{padded[i:i + 3]}", w) for i in range(len(padded) - 2)]
        return feats

    def embed(self, text: str) -> np.ndarray:
        feats = self._features(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        if not feats:
            return vec
        hashes = np.fromiter((_stable_hash(f) for f, _ in feats), dtype=np.uint64, count=len(feats))
        weights = np.fromiter((w for _, w in feats), dtype=np.float32, count=len(feats))
        idx = (hashes & np.uint64(self.dim - 1)).astype(np.int64)      # low bits -> position
        sign = np.where((hashes >> np.uint64(63)) & np.uint64(1), -1.0, 1.0).astype(np.float32)  # top bit -> sign
        np.add.at(vec, idx, sign * weights)  # scatter-add handles repeated indices correctly
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def __repr__(self) -> str:
        return f"HashingEmbedder(dim={self.dim})"


class SentenceTransformerEmbedder(Embedder):  # pragma: no cover - optional heavy dependency
    """Optional neural embedder. ``pip install sentence-transformers`` to use.

    Higher paraphrase recall than hashing, at the cost of a model download and
    ~5-20 ms per embedding on CPU.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("pip install sentence-transformers to use this embedder") from exc
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        return self.embed_many([text])[0]

    def embed_many(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity of the rows of ``a`` and ``b``.

    For unit vectors cosine similarity is just the dot product, so the whole
    (n x m) table is one matrix multiplication: ``a @ b.T``.
    """
    a_n = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b_n = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a_n @ b_n.T
