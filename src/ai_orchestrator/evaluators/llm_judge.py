from __future__ import annotations

import json
import logging
import re

from ai_orchestrator.evaluators.base import BaseEvaluator
from ai_orchestrator.models import EvaluationResult
from ai_orchestrator.prompts import render_llm_judge_prompt
from ai_orchestrator.providers import ProviderFactory
from ai_orchestrator.config.settings import settings

logger = logging.getLogger(__name__)

# Dimensions the judge scores and their pass thresholds.
_DIMENSIONS: dict[str, float] = {
    "correctness": 0.7,
    "completeness": 0.7,
    "groundedness": 0.7,
    "helpfulness": 0.7,
}

_JSON_PATTERN = re.compile(r"\{[^{}]+\}", re.DOTALL)


def _parse_scores(raw: str) -> dict[str, float]:
    """
    Extract the JSON scores object from the LLM response.

    The model is instructed to reply with bare JSON, but may wrap it in
    markdown fences or add surrounding text.  We use a regex to find the
    first ``{...}`` block and parse it.
    """
    match = _JSON_PATTERN.search(raw)
    if not match:
        raise ValueError(f"No JSON object found in judge response: {raw!r}")

    data = json.loads(match.group())

    scores: dict[str, float] = {}
    for dim in _DIMENSIONS:
        raw_val = data.get(dim)
        if raw_val is None:
            raise ValueError(f"Missing dimension '{dim}' in judge response: {data}")
        scores[dim] = max(0.0, min(1.0, float(raw_val)))

    return scores


class LlmJudgeEvaluator(BaseEvaluator):
    """
    LLM-as-a-Judge evaluator.

    Makes a single LLM call per test and returns four ``EvaluationResult``
    objects — one per quality dimension:

    - **correctness**   — Is the answer factually correct per the context?
    - **completeness**  — Does it address all key aspects of the question?
    - **groundedness**  — Is every claim supported by the context?
    - **helpfulness**   — Would a user find the answer useful?

    This complements the deterministic heuristic evaluators (hallucination,
    relevance, faithfulness) with an LLM perspective that can catch nuanced
    quality issues the keyword-based evaluators miss.

    The judge uses the same Gemini provider as the main answer generation,
    but with a structured JSON prompt so scores are machine-parseable.

    Usage:
        Add ``llm_judge`` to the ``evaluators`` list in ``execution.yaml``
        or a comparison run config.  One additional LLM call is made per
        test case.

    Graceful degradation:
        If the LLM call fails or returns unparseable JSON, each dimension
        gets score 0.0 with the error as the reason — the run does not
        crash.
    """

    def __init__(self) -> None:
        self._provider = ProviderFactory.create(settings.provider)

    def evaluate(
        self,
        test_id: str,
        question: str,
        answer: str,
        context: str,
    ) -> list[EvaluationResult]:
        prompt = render_llm_judge_prompt(
            question=question,
            context=context,
            answer=answer,
        )

        try:
            # We call ask() with the judge prompt as the "question" and an
            # empty context string so the provider wraps it in the right way.
            # Because the judge prompt is self-contained, context="" is fine.
            raw = self._provider.ask(prompt, "")
            scores = _parse_scores(raw)
            logger.debug("LLM judge scores for %s: %s", test_id, scores)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM judge failed for test %s: %s", test_id, exc)
            return [
                EvaluationResult(
                    test_id=test_id,
                    metric=f"llm_judge_{dim}",
                    score=0.0,
                    passed=False,
                    reason=f"Judge error: {exc}",
                )
                for dim in _DIMENSIONS
            ]

        return [
            EvaluationResult(
                test_id=test_id,
                metric=f"llm_judge_{dim}",
                score=round(score, 2),
                passed=score >= threshold,
                reason=(
                    f"LLM judge {dim}: {score:.2f} "
                    f"({'pass' if score >= threshold else 'fail'})"
                ),
            )
            for dim, (threshold, score) in zip(
                _DIMENSIONS.keys(),
                [(t, scores[d]) for d, t in _DIMENSIONS.items()],
            )
        ]
