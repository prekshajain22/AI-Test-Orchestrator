from abc import ABC, abstractmethod

from ai_orchestrator.models import EvaluationResult


class BaseEvaluator(ABC):
    """
    Base interface for all AI quality evaluators.
    """

    @abstractmethod
    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """
        Evaluate an AI response.
        """
        pass