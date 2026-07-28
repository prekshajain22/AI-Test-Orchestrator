from ai_orchestrator.evaluators.base import BaseEvaluator
from ai_orchestrator.models import EvaluationResult


class HallucinationEvaluator(BaseEvaluator):
    """
    Evaluates whether an AI response contains information
    that is not supported by the source document.
    """

    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """
        Compare AI response against source context.

        Returns:
            EvaluationResult containing hallucination score and explanation.
        """

        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        unsupported_words = answer_words - context_words

        score = 1.0 if not unsupported_words else 0.0

        passed = score == 1.0

        if passed:
            reason = (
                "Response is supported by the source document."
            )
        else:
            reason = (
                f"Potential unsupported information detected: "
                f"{unsupported_words}"
            )

        return EvaluationResult(
            test_id=test_id,
            metric="hallucination",
            score=score,
            passed=passed,
            reason=reason,
        )