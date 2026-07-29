from dataclasses import dataclass


@dataclass
class AIRequest:
    """Represents a question and context sent to an AI provider."""

    question: str
    context: str
