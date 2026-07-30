from unittest.mock import MagicMock, patch

import pytest

from ai_orchestrator.models.prompt_test_case import PromptTestCase
from ai_orchestrator.providers.base import ProviderRateLimitError
from ai_orchestrator.services.execution_service import ExecutionService


def _make_service_with_mocks(provider_mock, evaluator_names=None):
    """
    Build an ExecutionService with the provider/factory patched out so we
    can control what provider.ask() does, without touching real config,
    network calls, or evaluator internals.
    """
    with patch(
        "ai_orchestrator.services.execution_service.ProviderFactory.create",
        return_value=provider_mock,
    ), patch(
        "ai_orchestrator.services.execution_service.EvaluationFactory.create_all",
        return_value=[],
    ):
        service = ExecutionService(
            provider_name="gemini",
            test_suites=[],
            evaluators=evaluator_names or [],
        )
    return service


def test_provider_rate_limit_error_does_not_produce_scored_answer(monkeypatch):
    """
    Regression test: when the provider raises ProviderRateLimitError,
    ExecutionService must record it as a provider error (TestExecutionResult
    with `error` set and no evaluations) — NOT run evaluators against an
    error string as if it were a real answer.
    """
    provider_mock = MagicMock()
    provider_mock.ask.side_effect = ProviderRateLimitError("quota exceeded")

    service = _make_service_with_mocks(provider_mock)

    # Bypass real file loading / document loading.
    test_case = PromptTestCase(
        id="t1",
        question="What is the leave policy?",
        expected_answer="",
        source_document="dummy.md",
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_prompt_tests",
        lambda path: [test_case],
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_document",
        lambda path: "some context",
    )

    # Spy on the evaluation engine to make sure it's never invoked.
    service.engine.evaluate = MagicMock()

    results = service._run_suite("dummy_suite.yaml")

    assert len(results) == 1
    result = results[0]

    assert result.error is not None
    assert "quota exceeded" in result.error
    assert result.evaluations == []
    assert result.passed is False

    service.engine.evaluate.assert_not_called()


def test_successful_answer_is_still_evaluated(monkeypatch):
    provider_mock = MagicMock()
    provider_mock.ask.return_value = "A real model answer."

    service = _make_service_with_mocks(provider_mock)

    test_case = PromptTestCase(
        id="t1",
        question="What is the leave policy?",
        expected_answer="",
        source_document="dummy.md",
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_prompt_tests",
        lambda path: [test_case],
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_document",
        lambda path: "some context",
    )

    from ai_orchestrator.models.evaluation import EvaluationResult

    fake_result = EvaluationResult(
        test_id="t1", metric="hallucination", score=0.9, passed=True, reason="ok"
    )
    service.engine.evaluate = MagicMock(return_value=[fake_result])

    results = service._run_suite("dummy_suite.yaml")

    assert len(results) == 1
    result = results[0]
    assert result.error is None
    assert result.answer == "A real model answer."
    assert result.evaluations == [fake_result]
    assert result.passed is True
