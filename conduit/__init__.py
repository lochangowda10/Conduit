"""Conduit - a self-hosted, OpenAI-compatible LLM gateway.

    from conduit import Gateway, GatewayConfig, ChatRequest
    from conduit.providers import SimulatedProvider

    gw = Gateway([SimulatedProvider("sim-small", tier="small")])
    print(gw.complete(ChatRequest.from_prompt("What is the capital of France?")).text)
"""

from .cache import ResponseCache
from .config import GatewayConfig
from .gateway import Gateway
from .ledger import CostLedger
from .types import ChatRequest, ChatResponse, Message, Usage

__version__ = "1.0.0"

__all__ = [
    "Gateway",
    "GatewayConfig",
    "ChatRequest",
    "ChatResponse",
    "Message",
    "Usage",
    "ResponseCache",
    "CostLedger",
    "__version__",
]
