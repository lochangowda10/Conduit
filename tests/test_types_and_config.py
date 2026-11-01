import json

import pytest

from conduit.config import GatewayConfig
from conduit.errors import ConfigError
from conduit.types import ChatRequest, ChatResponse, Message, Usage


class TestUsage:
    def test_add_and_len(self):
        total = Usage(10, 5, 0.001) + Usage(20, 5, 0.002)
        assert total == Usage(30, 10, 0.003)
        assert len(total) == 40

    def test_sum_uses_radd(self):
        assert sum([Usage(1, 1, 0.1), Usage(2, 2, 0.2)]) == Usage(3, 3, 0.3)

    def test_repr_str_bool_lt(self):
        u = Usage(100, 50, 0.002)
        assert "prompt_tokens=100" in repr(u)
        assert str(u) == "150 tokens ($0.002000)"
        assert not Usage()
        assert Usage(0, 0, 0.001) < u

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            Usage(-1, 0, 0)

    def test_add_wrong_type(self):
        with pytest.raises(TypeError):
            Usage() + 5


class TestChatRequest:
    def test_from_openai(self):
        req = ChatRequest.from_openai({"model": "small", "messages": [
            {"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}], "temperature": 0.2})
        assert req.model == "small" and req.prompt_text == "hi" and req.system_text == "be brief"

    @pytest.mark.parametrize("payload", [{}, {"messages": "nope"}, {"messages": [{"role": "user"}]}])
    def test_from_openai_malformed(self, payload):
        with pytest.raises((ValueError, TypeError)):
            ChatRequest.from_openai(payload)

    def test_validation(self):
        with pytest.raises(ValueError):
            ChatRequest.from_prompt("x", temperature=3)
        with pytest.raises(ValueError):
            Message("robot", "x")

    def test_response_openai_shape(self):
        r = ChatResponse("hello", "p", "m", Usage(1, 2, 0.5), 12.3, cache="semantic", request_id="abc")
        out = r.to_openai()
        assert out["choices"][0]["message"]["content"] == "hello"
        assert out["usage"]["total_tokens"] == 3
        assert out["x_conduit"]["cache"] == "semantic"


class TestConfig:
    def test_defaults_valid(self):
        cfg = GatewayConfig()
        assert cfg.similarity_threshold == 0.85 and cfg.hedge_after_ms is None

    @pytest.mark.parametrize("key,value", [
        ("similarity_threshold", 1.5), ("similarity_threshold", "high"), ("max_attempts_per_provider", 0),
        ("rate_limit_rps", -1), ("budget_usd", -5), ("cache_capacity", True),
    ])
    def test_property_validation(self, key, value):
        with pytest.raises(ConfigError):
            GatewayConfig(**{key: value})

    def test_unknown_key(self):
        with pytest.raises(ConfigError, match="unknown config keys"):
            GatewayConfig(similarity=0.9)

    def test_from_json_nested_sections(self, tmp_path):
        p = tmp_path / "c.json"
        p.write_text(json.dumps({"_comment": "x", "cache": {"similarity_threshold": 0.9},
                                 "resilience": {"hedge_after_ms": 500}, "providers": []}))
        cfg = GatewayConfig.from_json(p)
        assert cfg.similarity_threshold == 0.9 and cfg.hedge_after_ms == 500

    def test_from_json_errors(self, tmp_path):
        with pytest.raises(ConfigError, match="not found"):
            GatewayConfig.from_json(tmp_path / "nope.json")
        bad = tmp_path / "bad.json"
        bad.write_text("{ not json")
        with pytest.raises(ConfigError, match="invalid JSON"):
            GatewayConfig.from_json(bad)

    def test_example_config_loads(self):
        from pathlib import Path
        cfg = GatewayConfig.from_json(Path(__file__).resolve().parent.parent / "config.example.json")
        assert len(cfg.providers) >= 1
