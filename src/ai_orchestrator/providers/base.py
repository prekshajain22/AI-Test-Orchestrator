from abc import ABC, abstractmethod


class ProviderError(Exception):
    """Base exception raised when an LLM provider fails to produce an answer."""


class ProviderRateLimitError(ProviderError):
    """Raised when a provider reports it is rate-limited / quota exceeded.

    This must NEVER be silently converted into a plain string answer,
    since evaluators would then score the error message as if it were
    a genuine model response, silently corrupting results.
    """


class LLMClient(ABC):

    @abstractmethod
    def ask(self, question: str, context: str) -> str:
        """Ask the language model a question."""
        pass
