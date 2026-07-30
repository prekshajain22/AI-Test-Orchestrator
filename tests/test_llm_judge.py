"""
Tests for LlmJudgeEvaluator.

All tests mock the underlying provider so no real API calls are made.
"""
from unittest.mock import MagicMock, patch

import pytest

from ai_orchestrator.evaluators.llm_judge import LlmJudgeEvaluator, _parse_scores
from ai_orchestrator.models import EvaluationResult


# ── _parse_scores unit tests ──────────────────────────────────────────────────

def test_parse_scores_valid_json():
    raw = '{"correctness": 0.9, "completeness": 0.8, "groundedness": 0.95, "helpfulness": 0.85}'
    scores = _parse_scores(raw)
    assert scores == pytest.approx(
        {"correctness": 0.9, "completeness": 0.8, "groundedness": 0.95, "helpfulness": 0.85}
    )


def test_parse_scores_handles_markdown_fences():
    raw = '```json\n{"correctness": 0.7, "completeness": 0.6, "groundedness": 0.8, "helpfulness": 0.75}\n```'
    scores = _parse_scores(raw)
    assert scores["correctness"] == pytest.approx(0.7)


def test_parse_scores_clamps_above_1():
    raw = '{"correctness": 1.5, "completeness": 0.8, "groundedness": 0.9, "helpfulness": 0.7}'
    scores = _parse_scores(raw)
    assert scores["correctness"] == 1.0


def test_parse_scores_clamps_below_0():
    raw = '{"correctness": -0.2, "completeness": 0.8, "groundedness": 0.9, "helpfulness": 0.7}'
    scores = _parse_scores(raw)
    assert scores["correctness"] == 0.0


def test_parse_scores_raises_on_no_json():
    with pytest.raises(ValueError, match="No JSON object found"):
        _parse_scores("Sorry, I cannot score this.")


def test_parse_scores_raises_on_missing_dimension():
    raw = '{"correctness": 0.9, "completeness": 0.8}'
    with pytest.raises(ValueError, match="Missing dimension"):
        _parse_scores(raw)


# ── LlmJudgeEvaluator integration tests (provider mocked) ────────────────────

def _make_judge(mock_response: str) -> LlmJudgeEvaluator:
    """Build an LlmJudgeEvaluator with the provider's ask() returning mock_response."""
    provider_mock = MagicMock()
    provider_mock.ask.return_value = mock_response

    with patch(
        "ai_orchestrator.providers.ProviderFactory.create",
        return_value=provider_mock,
    ):
        judge = LlmJudgeEvaluator()

    return judge


def test_llm_judge_returns_four_evaluation_results():
    judge = _make_judge(
        '{"correctness": 0.9, "completeness": 0.8, "groundedness": 0.95, "helpfulness": 0.85}'
    )
    results = judge.evaluate(
        test_id="t1",
        question="What is the leave policy?",
        answer="Employees get 25 days annual leave.",
        context="Employees are entitled to 25 days of annual leave per year.",
    )
    assert len(results) == 4


def test_llm_judge_result_metrics_are_named_correctly():
    judge = _make_judge(
        '{"correctness": 0.9, "completeness": 0.8, "groundedness": 0.95, "helpfulness": 0.85}'
    )
    results = judge.evaluate(test_id="t1", question="Q", answer="A", context="C")
    metrics = {r.metric for r in results}
    assert metrics == {
        "llm_judge_correctness",
        "llm_judge_completeness",
        "llm_judge_groundedness",
        "llm_judge_helpfulness",
    }


def test_llm_judge_scores_match_parsed_values():
    judge = _make_judge(
        '{"correctness": 0.9, "completeness": 0.8, "groundedness": 0.95, "helpfulness": 0.85}'
    )
    results = judge.evaluate(test_id="t1", question="Q", answer="A", context="C")
    by_metric = {r.metric: r for r in results}
    assert by_metric["llm_judge_correctness"].score == pytest.approx(0.9)
    assert by_metric["llm_judge_completeness"].score == pytest.approx(0.8)
    assert by_metric["llm_judge_groundedness"].score == pytest.approx(0.95)
    assert by_metric["llm_judge_helpfulness"].score == pytest.approx(0.85)


def test_llm_judge_passes_when_score_above_threshold():
    judge = _make_judge(
        '{"correctness": 0.9, "completeness": 0.8, "groundedness": 0.95, "helpfulness": 0.85}'
    )
    results = judge.evaluate(test_id="t1", question="Q", answer="A", context="C")
    assert all(r.passed for r in results)


def test_llm_judge_fails_when_score_below_threshold():
    judge = _make_judge(
        '{"correctness": 0.4, "completeness": 0.3, "groundedness": 0.5, "helpfulness": 0.2}'
    )
    results = judge.evaluate(test_id="t1", question="Q", answer="A", context="C")
    assert all(not r.passed for r in results)


def test_llm_judge_graceful_degradation_on_provider_error():
    """When the provider raises, all 4 dimensions should get score=0 with error reason."""
    provider_mock = MagicMock()
    provider_mock.ask.side_effect = RuntimeError("API unavailable")

    with patch(
        "ai_orchestrator.providers.ProviderFactory.create",
        return_value=provider_mock,
    ):
        judge = LlmJudgeEvaluator()

    results = judge.evaluate(test_id="t1", question="Q", answer="A", context="C")

    assert len(results) == 4
    assert all(r.score == 0.0 for r in results)
    assert all(not r.passed for r in results)
    assert all("Judge error" in r.reason for r in results)


def test_llm_judge_graceful_degradation_on_unparseable_response():
    """Unparseable JSON from the model should not crash the run."""
    judge = _make_judge("I cannot evaluate this answer.")
    results = judge.evaluate(test_id="t1", question="Q", answer="A", context="C")
    assert len(results) == 4
    assert all(r.score == 0.0 for r in results)


def test_llm_judge_all_results_are_evaluation_result_instances():
    judge = _make_judge(
        '{"correctness": 0.9, "completeness": 0.8, "groundedness": 0.95, "helpfulness": 0.85}'
    )
    results = judge.evaluate(test_id="t1", question="Q", answer="A", context="C")
    assert all(isinstance(r, EvaluationResult) for r in results)


def test_factory_creates_llm_judge():
    """EvaluationFactory must recognise the 'llm_judge' name."""
    from ai_orchestrator.evaluators import EvaluationFactory

    with patch(
        "ai_orchestrator.providers.ProviderFactory.create",
        return_value=MagicMock(),
    ):
        evaluator = EvaluationFactory.create("llm_judge")

    assert isinstance(evaluator, LlmJudgeEvaluator)
