import os

from ..core.model import Model
from ..env import load_env_upwards
from .anthropic import AnthropicModel
from .openai import OpenAIModel


def model(
    provider: str | None = None,
    *,
    name: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: float | None = None,
    retries: int | None = None,
    retry_delay: float | None = None,
    opts: dict | None = None,
) -> Model:
    """Build a provider model from explicit args or environment variables."""
    load_env_upwards()
    chosen = (provider or _default_provider()).strip().lower()
    if chosen in {"anthropic", "claude"}:
        return AnthropicModel(
            api_key=api_key,
            base_url=base_url,
            model=name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            opts=opts,
        )
    if chosen in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAIModel(
            api_key=api_key,
            base_url=base_url,
            model=name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
            opts=opts,
        )
    raise ValueError(f"Unknown model provider: {chosen}")


def _default_provider() -> str:
    provider = os.getenv("MINIADK_MODEL_PROVIDER")
    if provider:
        return provider
    if _has_env("MINIADK_MODEL_KEY"):
        if _looks_anthropic_url(os.getenv("MINIADK_MODEL_URL", "")):
            return "anthropic"
        return "openai"
    if _has_env("ANTHROPIC_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        return "anthropic"
    if _has_env("OPENAI_KEY", "OPENAI_API_KEY"):
        return "openai"
    raise ValueError(
        "model() requires provider, MINIADK_MODEL_PROVIDER, or ANTHROPIC_KEY/ANTHROPIC_API_KEY/OPENAI_KEY/OPENAI_API_KEY"
    )


def _has_env(*names: str) -> bool:
    return any(os.getenv(name) for name in names)


def _looks_anthropic_url(url: str) -> bool:
    text = url.lower()
    return "anthropic" in text or "claude" in text
