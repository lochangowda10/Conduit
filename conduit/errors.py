"""Exception hierarchy.

A small, explicit hierarchy lets the resilience layer decide *what* to do
with a failure just by its type:

* ``RetryableError``  -> retry on the same provider (with backoff), then fall back
* ``ProviderError``   -> do not retry here, but another provider may succeed
* ``ClientError``     -> the request itself is bad; no provider will fix it
"""


class ConduitError(Exception):
    """Base class for every error raised by Conduit."""


class ConfigError(ConduitError):
    """Configuration file is missing, malformed or contains invalid values."""


class ClientError(ConduitError):
    """The caller sent something invalid (HTTP 400). Never retried."""


class ProviderError(ConduitError):
    """A provider failed. Subclasses say whether a retry might help."""

    status_code = 502

    def __init__(self, message: str, provider: str = "?"):
        super().__init__(f"[{provider}] {message}")
        self.provider = provider


class RetryableError(ProviderError):
    """Transient failure: worth retrying after a short wait."""


class RateLimitError(RetryableError):
    """Provider said HTTP 429 Too Many Requests."""

    status_code = 429

    def __init__(self, message: str, provider: str = "?", retry_after: float | None = None):
        super().__init__(message, provider)
        self.retry_after = retry_after


class ProviderTimeout(RetryableError):
    """Provider did not answer within the configured timeout."""

    status_code = 504


class ProviderUnavailable(RetryableError):
    """HTTP 5xx or connection error."""

    status_code = 503


class ProviderBadRequest(ProviderError):
    """Provider rejected the request (4xx other than 429). Retrying is pointless."""

    status_code = 400


class CircuitOpenError(ProviderError):
    """The circuit breaker for this provider is open; we skip it without calling."""

    status_code = 503


class AllProvidersFailed(ConduitError):
    """Every provider in the fallback chain failed."""

    status_code = 503

    def __init__(self, errors: list[Exception]):
        self.errors = errors
        detail = "; ".join(f"{type(e).__name__}: {e}" for e in errors) or "no eligible providers"
        super().__init__(f"all providers failed -> {detail}")


class RateLimitedByGateway(ConduitError):
    """The gateway's own token bucket rejected the request."""

    status_code = 429


class BudgetExceeded(ConduitError):
    """The configured spending cap has been reached (cache hits are still served)."""

    status_code = 402
