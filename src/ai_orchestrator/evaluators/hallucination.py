from ai_orchestrator.models import EvaluationResult


class HallucinationEvaluator:
    """
    Evaluates whether an AI response contains information
    not supported by the source document.
    """

    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:

        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())

        unsupported_words = answer_words - context_words

        score = 1.0

        if unsupported_words:
            score = 0.0

        passed = score == 1.0

        reason = (
            "Response is supported by the source document."
            if passed
            else f"Potential unsupported information detected: {unsupported_words}"
        )

        return EvaluationResult(
            test_id=test_id,
            metric="hallucination",
            score=score,
            passed=passed,
            reason=reason,
        )