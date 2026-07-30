from ai_orchestrator.providers.base import (
    LLMClient,
    ProviderError,
    ProviderRateLimitError,
)


class ProviderRegistry:
    """Maps provider names to their implementation classes."""

    @classmethod
    def get(cls, name: str) -> type[LLMClient]:
        if name == "gemini":
            from ai_orchestrator.providers.gemini import GeminiProvider
            return GeminiProvider
        if name == "huggingface":
            from ai_orchestrator.providers.huggingface import HuggingFaceClient
            return HuggingFaceClient
        raise ValueError(f"Unknown provider: '{name}'")


class ProviderFactory:
    """Creates LLM provider instances by name."""

    @classmethod
    def create(cls, provider_name: str) -> LLMClient:
        return ProviderRegistry.get(provider_name)()


__all__ = [
    "LLMClient",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderRegistry",
    "ProviderFactory",
]
