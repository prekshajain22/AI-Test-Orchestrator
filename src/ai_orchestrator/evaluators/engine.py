from ai_orchestrator.evaluators.base import BaseEvaluator
from ai_orchestrator.models import EvaluationResult


class EvaluationEngine:
    """
    Executes one or more evaluators against an AI response.
    """

    def __init__(self):
        self._evaluators: list[BaseEvaluator] = []

    def register(self, evaluator: BaseEvaluator) -> None:
        """
        Register an evaluator.
        """
        self._evaluators.append(evaluator)

    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> list[EvaluationResult]:
        """
        Execute all registered evaluators.
        """

        results = []

        for evaluator in self._evaluators:
            result = evaluator.evaluate(
                test_id=test_id,
                question=question,
                answer=answer,
                context=context,
            )

            results.append(result)

        return results