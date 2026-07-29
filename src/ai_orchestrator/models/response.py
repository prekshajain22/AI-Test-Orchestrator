from dataclasses import dataclass


@dataclass
class AIResponse:
    """Represents an answer returned by an AI provider."""

    answer: str
