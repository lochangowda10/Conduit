"""Core data types that flow through the gateway.

Syllabus link
-------------
* Experiment 5 (Magic Methods): ``Usage`` implements ``__add__``, ``__radd__``,
  ``__eq__``, ``__len__``, ``__bool__``, ``__str__`` and ``__repr__`` so that
  token/cost accounting reads like arithmetic: ``total = sum(usages)``.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Message:
    """One chat message in the OpenAI format: ``{"role": ..., "content": ...}``."""

    role: str
    content: str

    VALID_ROLES = ("system", "user", "assistant", "tool")

    def __post_init__(self) -> None:
        if self.role not in self.VALID_ROLES:
            raise ValueError(f"invalid role {self.role!r}; expected one of {self.VALID_ROLES}")
        if not isinstance(self.content, str):
            raise TypeError("message content must be a string")

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class ChatRequest:
    """A provider-agnostic chat completion request.

    ``model`` is a *logical* name ("auto", "small", "large" or a concrete
    provider name). The router decides which physical provider serves it.
    """

    messages: list[Message]
    model: str = "auto"
    temperature: float = 0.0
    max_tokens: int = 256
    stream: bool = False
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.messages:
            raise ValueError("a request needs at least one message")
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be in [0, 2]")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")

    @classmethod
    def from_openai(cls, payload: dict[str, Any]) -> "ChatRequest":
        """Alternate constructor: build from an OpenAI-style JSON body."""
        try:
            messages = [Message(m["role"], m["content"]) for m in payload["messages"]]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed 'messages' field: {exc}") from exc
        return cls(
            messages=messages,
            model=payload.get("model", "auto"),
            temperature=float(payload.get("temperature", 0.0)),
            max_tokens=int(payload.get("max_tokens", 256)),
            stream=bool(payload.get("stream", False)),
        )

    @classmethod
    def from_prompt(cls, prompt: str, **kwargs: Any) -> "ChatRequest":
        """Alternate constructor for the common single-user-message case."""
        return cls(messages=[Message("user", prompt)], **kwargs)

    @property
    def prompt_text(self) -> str:
        """The last user message — what the cache and router look at."""
        for msg in reversed(self.messages):
            if msg.role == "user":
                return msg.content
        return self.messages[-1].content

    @property
    def system_text(self) -> str:
        return "\n".join(m.content for m in self.messages if m.role == "system")

    @property
    def all_text(self) -> str:
        return "\n".join(m.content for m in self.messages)


class Usage:
    """Token and cost accounting with operator overloading.

    >>> a = Usage(10, 5, 0.001)
    >>> b = Usage(20, 5, 0.002)
    >>> a + b
    Usage(prompt_tokens=30, completion_tokens=10, cost_usd=0.003)
    >>> len(a + b)          # total tokens
    40
    >>> sum([a, b]) == a + b  # works because of __radd__
    True
    """

    __slots__ = ("prompt_tokens", "completion_tokens", "cost_usd")

    def __init__(self, prompt_tokens: int = 0, completion_tokens: int = 0, cost_usd: float = 0.0):
        if prompt_tokens < 0 or completion_tokens < 0 or cost_usd < 0:
            raise ValueError("usage values cannot be negative")
        self.prompt_tokens = int(prompt_tokens)
        self.completion_tokens = int(completion_tokens)
        self.cost_usd = float(cost_usd)

    # --- arithmetic -------------------------------------------------------
    def __add__(self, other: object) -> "Usage":
        if not isinstance(other, Usage):
            return NotImplemented
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            round(self.cost_usd + other.cost_usd, 10),
        )

    def __radd__(self, other: object) -> "Usage":
        # sum() starts from the integer 0, so 0 + Usage must work.
        if other == 0:
            return self
        return self.__add__(other)

    # --- comparison -------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Usage):
            return NotImplemented
        return (
            self.prompt_tokens == other.prompt_tokens
            and self.completion_tokens == other.completion_tokens
            and abs(self.cost_usd - other.cost_usd) < 1e-9
        )

    __hash__ = None  # mutable value object -> deliberately unhashable

    def __lt__(self, other: "Usage") -> bool:
        if not isinstance(other, Usage):
            return NotImplemented
        return self.cost_usd < other.cost_usd

    # --- container-ish / truthiness ---------------------------------------
    def __len__(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __bool__(self) -> bool:
        return len(self) > 0 or self.cost_usd > 0

    # --- representation ---------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Usage(prompt_tokens={self.prompt_tokens}, "
            f"completion_tokens={self.completion_tokens}, cost_usd={self.cost_usd:g})"
        )

    def __str__(self) -> str:
        return f"{len(self):,} tokens (${self.cost_usd:.6f})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": len(self),
            "cost_usd": self.cost_usd,
        }


@dataclass
class ChatResponse:
    """What the gateway returns to callers."""

    text: str
    provider: str
    model: str
    usage: Usage
    latency_ms: float
    cache: str = "miss"  # "miss" | "exact" | "semantic"
    attempts: int = 1
    hedged: bool = False
    request_id: str = ""
    created: float = field(default_factory=time.time)

    def to_openai(self) -> dict[str, Any]:
        """Serialise into the OpenAI ``chat.completion`` response shape."""
        return {
            "id": f"chatcmpl-{self.request_id}",
            "object": "chat.completion",
            "created": int(self.created),
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": self.usage.prompt_tokens,
                "completion_tokens": self.usage.completion_tokens,
                "total_tokens": len(self.usage),
            },
            # Extension field: tells the caller what the gateway did.
            "x_conduit": {
                "provider": self.provider,
                "cache": self.cache,
                "latency_ms": round(self.latency_ms, 2),
                "attempts": self.attempts,
                "hedged": self.hedged,
                "cost_usd": self.usage.cost_usd,
            },
        }


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 characters per token for English).

    Real tokenizers (tiktoken, sentencepiece) are exact but model-specific;
    for routing and budgeting an estimate is good enough and dependency-free.
    """
    return max(1, (len(text) + 3) // 4)
