import re

from ai_orchestrator.evaluators.base import BaseEvaluator
from ai_orchestrator.models import EvaluationResult


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Evaluates whether the AI answer faithfully represents
    the information contained in the source document.
    """

    PASS_THRESHOLD = 0.70

    STOP_WORDS = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "to",
        "of",
        "and",
        "or",
        "in",
        "on",
        "at",
        "for",
        "with",
        "by",
        "as",
        "that",
        "this",
        "it",
        "its",
        "their",
        "there",
        "from",
        "into",
        "about",
        "before",
        "after",
        "than",
        "may",
        "must",
        "should",
        "can",
        "could",
        "would",
        "will",
        "based",
        "provided",
        "context",
        "according",
        "information",
    }

    def _normalize(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s.]", "", text)
        return text

    def _extract_keywords(self, text: str) -> set[str]:
        words = self._normalize(text).split()

        return {
            word
            for word in words
            if len(word) > 2 and word not in self.STOP_WORDS
        }

    def _split_sentences(self, text: str) -> list[str]:
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _sentence_similarity(
        self,
        answer_sentence: str,
        context_sentence: str,
    ) -> float:

        answer_words = self._extract_keywords(answer_sentence)
        context_words = self._extract_keywords(context_sentence)

        if not answer_words:
            return 0.0

        common = answer_words & context_words

        return len(common) / len(answer_words)

    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> EvaluationResult:

        answer_sentences = self._split_sentences(answer)
        context_sentences = self._split_sentences(context)

        if not answer_sentences:
            return EvaluationResult(
                test_id=test_id,
                metric="faithfulness",
                score=0.0,
                passed=False,
                reason="Answer is empty.",
            )

        similarities = []

        for answer_sentence in answer_sentences:

            best_score = 0.0

            for context_sentence in context_sentences:
                score = self._sentence_similarity(
                    answer_sentence,
                    context_sentence,
                )

                best_score = max(best_score, score)

            similarities.append(best_score)

        final_score = round(
            sum(similarities) / len(similarities),
            2,
        )

        passed = final_score >= self.PASS_THRESHOLD

        if passed:
            reason = "Answer is faithful to the source document."
        else:
            reason = (
                "Some statements in the answer are weakly "
                "supported by the source document."
            )

        return EvaluationResult(
            test_id=test_id,
            metric="faithfulness",
            score=final_score,
            passed=passed,
            reason=reason,
        )