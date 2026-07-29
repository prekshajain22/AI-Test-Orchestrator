from ai_orchestrator.providers.registry import ProviderRegistry
from ai_orchestrator.providers.base import LLMClient


class ProviderFactory:
    """Creates LLM provider instances by name using the provider registry."""

    @classmethod
    def create(cls, provider_name: str) -> LLMClient:
        return ProviderRegistry.get(provider_name)()
