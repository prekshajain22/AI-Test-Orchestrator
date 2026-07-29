from ai_orchestrator.providers.gemini import GeminiProvider
from ai_orchestrator.providers.huggingface import HuggingFaceClient


class ProviderRegistry:
    """Maps provider names to their implementation classes."""

    _registry = {
        "gemini": GeminiProvider,
        "huggingface": HuggingFaceClient,
    }

    @classmethod
    def get(cls, name: str):
        if name not in cls._registry:
            raise ValueError(f"Unknown provider: '{name}'")
        return cls._registry[name]
