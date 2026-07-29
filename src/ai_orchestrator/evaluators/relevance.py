import re

from ai_orchestrator.evaluators.base import BaseEvaluator
from ai_orchestrator.models import EvaluationResult


class RelevanceEvaluator(BaseEvaluator):
    """
    Evaluates whether the AI response is relevant
    to the user's question.
    """

    STOP_WORDS = {
        "what",
        "when",
        "where",
        "who",
        "how",
        "is",
        "are",
        "the",
        "a",
        "an",
        "to",
        "of",
        "for",
        "and",
        "can",
        "does",
        "do",
        "while",
        "working",
        "should",
        "be",
    }

    SYNONYMS = {
        "handled": {
            "handled",
            "protected",
            "managed",
            "secured",
        },
        "remotely": {
            "remotely",
            "remote",
        },
        "working": {
            "working",
            "work",
        },
        "required": {
            "required",
            "needed",
            "necessary",
        },
        "notify": {
            "notify",
            "inform",
            "tell",
        },
    }

    def _extract_keywords(self, text: str) -> set[str]:
        """
        Extract meaningful keywords from text.
        """

        text = text.lower()

        text = re.sub(r"[^\w\s]", "", text)

        words = set(text.split())

        return words - self.STOP_WORDS


    def _expand_keywords(self, keywords: set[str]) -> set[str]:
        """
        Expand keywords using simple domain synonyms.
        """

        expanded = set(keywords)

        for word in keywords:
            if word in self.SYNONYMS:
                expanded.update(
                    self.SYNONYMS[word]
                )

        return expanded


    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:
        """
        Compare question intent with AI response.
        """

        question_keywords = self._extract_keywords(
            question
        )

        answer_keywords = self._expand_keywords(
            self._extract_keywords(answer)
        )


        matched_keywords = question_keywords.intersection(
            answer_keywords
        )


        score = (
            len(matched_keywords)
            /
            max(len(question_keywords), 1)
        )


        passed = score >= 0.5


        reason = (
            "Answer is relevant to the question."
            if passed
            else (
                "Answer may not address question. "
                f"Missing keywords: "
                f"{question_keywords - answer_keywords}"
            )
        )


        return EvaluationResult(
            test_id=test_id,
            metric="relevance",
            score=round(score, 2),
            passed=passed,
            reason=reason,
        )