import pytest
from unittest.mock import MagicMock, patch

from ai_orchestrator.providers import ProviderFactory, ProviderRegistry
from ai_orchestrator.providers.base import LLMClient


# ── ProviderRegistry ───────────────────────────────────────────

def test_registry_returns_gemini_class():
    cls = ProviderRegistry.get("gemini")
    from ai_orchestrator.providers.gemini import GeminiProvider
    assert cls is GeminiProvider


def test_registry_raises_for_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        ProviderRegistry.get("nonexistent")


# ── LLMProviderFactory ─────────────────────────────────────────

def test_factory_create_returns_llm_client_instance():
    with patch("ai_orchestrator.providers.gemini.GeminiProvider.__init__", return_value=None):
        provider = ProviderFactory.create("gemini")
    assert isinstance(provider, LLMClient)


def test_factory_create_raises_for_unknown_provider():
    with pytest.raises(ValueError):
        ProviderFactory.create("unknown_provider")


# ── LLMClient contract ─────────────────────────────────────────

def test_llm_client_ask_returns_string():
    """Any provider returned by the factory must implement ask() → str."""
    mock_provider = MagicMock(spec=LLMClient)
    mock_provider.ask.return_value = "mocked answer"

    result = mock_provider.ask("What is the leave policy?", "context text")
    assert isinstance(result, str)
    assert result == "mocked answer"
