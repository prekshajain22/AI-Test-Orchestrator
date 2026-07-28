from abc import ABC, abstractmethod


class LLMClient(ABC):

    @abstractmethod
    def ask(self, question: str, context: str) -> str:
        """Ask the language model a question."""
        pass