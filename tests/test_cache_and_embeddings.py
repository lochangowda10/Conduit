import threading

import numpy as np
import pytest

from conduit.cache import ResponseCache, salient_tokens
from conduit.embeddings import HashingEmbedder, content_tokens, cosine_similarity_matrix
from conduit.types import ChatRequest, ChatResponse, Usage


def resp(text):
    return ChatResponse(text, "p", "m", Usage(1, 1, 0.001), 10.0)


class TestEmbeddings:
    def test_unit_norm_and_deterministic(self):
        e = HashingEmbedder(256)
        a, b = e.embed("hello world"), e.embed("hello world")
        assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)

    def test_paraphrase_closer_than_unrelated(self):
        e = HashingEmbedder()
        q = e.embed("What is the capital of France?")
        assert q @ e.embed("Tell me the capital of France.") > q @ e.embed("Explain recursion simply.")

    def test_empty_text(self):
        assert not HashingEmbedder(64).embed("the of a").any()  # only stopwords -> zero vector

    def test_bad_dim(self):
        with pytest.raises(ValueError):
            HashingEmbedder(100)

    def test_content_tokens(self):
        assert content_tokens("What's France's capital?") == ["france", "capital"]

    def test_similarity_matrix(self):
        m = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
        s = cosine_similarity_matrix(m, m)
        assert s.shape == (3, 3) and np.allclose(np.diag(s), 1.0) and np.isclose(s[0, 1], 0.0)


class TestSalientTokens:
    def test_numbers_and_proper_nouns(self):
        assert salient_tokens("What is 12 times 7?") == {"12", "7"}
        assert salient_tokens("Hey, Which city is the capital of Egypt? Thanks!") == {"egypt"}
        assert salient_tokens("France's capital is which city?") == {"france"}

    def test_quotes(self):
        assert "good morning" in salient_tokens('Translate "good morning" to French.')


class TestResponseCache:
    def test_exact_hit(self):
        c = ResponseCache()
        req = ChatRequest.from_prompt("What is the capital of France?")
        assert c.get(req) is None
        c.put(req, resp("Paris"))
        entry, kind, sim = c.get(ChatRequest.from_prompt("What is the capital of France?"))
        assert kind == "exact" and entry.response_text == "Paris" and sim == 1.0

    def test_semantic_hit_and_entity_guard(self):
        c = ResponseCache(similarity_threshold=0.8)
        c.put(ChatRequest.from_prompt("What is the capital of France?"), resp("Paris"))
        hit = c.get(ChatRequest.from_prompt("Tell me the capital of France."))
        assert hit is not None and hit[1] == "semantic"
        assert c.get(ChatRequest.from_prompt("What is the capital of Spain?")) is None

    def test_guard_blocks_number_change_even_at_low_threshold(self):
        c = ResponseCache(similarity_threshold=0.3, idf_weighting=False)
        c.put(ChatRequest.from_prompt("What is 12 times 7?"), resp("84"))
        assert c.get(ChatRequest.from_prompt("What is 12 times 8?")) is None
        assert c.stats.guard_rejections >= 1

    def test_scope_isolation(self):
        c = ResponseCache()
        a = ChatRequest.from_prompt("hello there friend", max_tokens=100)
        c.put(a, resp("x"))
        assert c.get(ChatRequest.from_prompt("hello there friend", max_tokens=500)) is None

    def test_high_temperature_not_cached(self):
        c = ResponseCache(max_temperature=0.3)
        req = ChatRequest.from_prompt("write a poem", temperature=1.0)
        c.put(req, resp("roses"))
        assert len(c) == 0 and c.get(req) is None

    def test_ttl_expiry(self, clock):
        c = ResponseCache(ttl_seconds=10, clock=clock)
        req = ChatRequest.from_prompt("cache me")
        c.put(req, resp("ok"))
        clock.advance(11)
        assert c.get(req) is None and c.stats.expirations == 1

    def test_lru_eviction(self):
        c = ResponseCache(capacity=2, semantic_enabled=False)
        r1, r2, r3 = (ChatRequest.from_prompt(f"question number {w}") for w in ("one", "two", "three"))
        c.put(r1, resp("1"))
        c.put(r2, resp("2"))
        c.get(r1)                   # r1 becomes most recently used
        c.put(r3, resp("3"))        # evicts r2
        assert r1 in c and r3 in c and r2 not in c and c.stats.evictions == 1

    def test_idf_weights_shape_and_order(self):
        c = ResponseCache()
        for word in ("alpha", "beta", "gamma"):
            c.put(ChatRequest.from_prompt(f"explain {word}"), resp(word))
        w = c.idf_weights()
        assert w.shape == (c.embedder.dim,) and w.min() >= 1.0

    def test_thread_safety(self):
        c = ResponseCache(capacity=500)

        def worker(k):
            for i in range(50):
                req = ChatRequest.from_prompt(f"thread {k} question {i}")
                c.put(req, resp(str(i)))
                c.get(req)

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(c) == 400 and c.stats.exact_hits == 400
