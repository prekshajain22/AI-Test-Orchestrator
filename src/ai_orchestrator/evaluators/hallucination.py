import re

from ai_orchestrator.evaluators.base import BaseEvaluator
from ai_orchestrator.models import EvaluationResult


class HallucinationEvaluator(BaseEvaluator):
    """
    Evaluates whether an AI response contains information
    unsupported by the source document.
    """

    IGNORE_WORDS = {
        "based",
        "provided",
        "context",
        "according",
        "information",
        "response",
        "answer",
        "following",
        "criteria",
        "there",
        "is",
        "are",
        "was",
        "were",
        "the",
        "a",
        "an",
        "to",
        "of",
        "and",
        "for",
        "in",
        "on",
        "with",
        "must",
        "should",
    }

    def _clean_text(self, text: str) -> set[str]:
        """
        Normalize text for comparison.
        """

        text = text.lower()

        text = re.sub(r"[^\w\s]", "", text)

        words = set(text.split())

        return words - self.IGNORE_WORDS


    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> list[EvaluationResult]:

        answer_words = self._clean_text(answer)
        context_words = self._clean_text(context)

        unsupported_words = answer_words - context_words

        # Ignore very small differences
        confidence = 1.0 - (
            len(unsupported_words) /
            max(len(answer_words), 1)
        )

        passed = confidence >= 0.7

        reason = (
            "Response appears supported by source document."
            if passed
            else f"Potential unsupported information detected: {unsupported_words}"
        )

        return [EvaluationResult(
            test_id=test_id,
            metric="hallucination",
            score=round(confidence, 2),
            passed=passed,
            reason=reason,
        )]
