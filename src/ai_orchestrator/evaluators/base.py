from abc import ABC, abstractmethod

from ai_orchestrator.models import EvaluationResult


class BaseEvaluator(ABC):
    """
    Base interface for all AI quality evaluators.

    ``evaluate()`` returns a *list* so that a single evaluator can produce
    multiple scored dimensions from one LLM call (e.g. LlmJudgeEvaluator
    returns four results — correctness, completeness, groundedness,
    helpfulness — while keeping the API call count to one).

    All deterministic heuristic evaluators return a single-item list.
    """

    @abstractmethod
    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> list[EvaluationResult]:
        """
        Evaluate an AI response.  Returns one or more EvaluationResult
        objects, each representing a distinct quality dimension.
        """
        pass
