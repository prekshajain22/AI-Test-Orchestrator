import pytest
from unittest.mock import MagicMock, patch

from ai_orchestrator.providers import ProviderFactory, ProviderRegistry
from ai_orchestrator.providers.base import LLMClient, ProviderRateLimitError


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


# ── GeminiProvider rate-limit handling ──────────────────────────

def test_gemini_ask_raises_on_429_instead_of_returning_string():
    """
    Regression test: a 429 rate-limit error must raise ProviderRateLimitError,
    NOT be returned as a plain string. Returning a string would let the
    error message get scored by evaluators as if it were a real answer,
    silently corrupting results.
    """
    from ai_orchestrator.providers.gemini import GeminiProvider
    from google.genai.errors import ClientError

    with patch("ai_orchestrator.providers.gemini.GeminiProvider.__init__", return_value=None):
        provider = GeminiProvider()

    mock_client = MagicMock()
    error = ClientError(code=429, response_json={"error": {"message": "quota exceeded"}})
    mock_client.models.generate_content.side_effect = error
    provider.client = mock_client
    provider.model = "gemini-test"

    with pytest.raises(ProviderRateLimitError):
        provider.ask("question", "context")


def test_gemini_ask_reraises_non_429_client_errors():
    from ai_orchestrator.providers.gemini import GeminiProvider
    from google.genai.errors import ClientError

    with patch("ai_orchestrator.providers.gemini.GeminiProvider.__init__", return_value=None):
        provider = GeminiProvider()

    mock_client = MagicMock()
    error = ClientError(code=500, response_json={"error": {"message": "server error"}})
    mock_client.models.generate_content.side_effect = error
    provider.client = mock_client
    provider.model = "gemini-test"

    with pytest.raises(ClientError):
        provider.ask("question", "context")
