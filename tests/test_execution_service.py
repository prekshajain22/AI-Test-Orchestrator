from unittest.mock import MagicMock, patch

import pytest

from ai_orchestrator.config.loader import RagConfig
from ai_orchestrator.models.prompt_test_case import PromptTestCase
from ai_orchestrator.providers.base import ProviderRateLimitError
from ai_orchestrator.services.execution_service import ExecutionService


def _make_service_with_mocks(provider_mock, evaluator_names=None, rag_config=None):
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
            rag_config=rag_config,
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


# ── RAG mode tests ────────────────────────────────────────────────────────────

def test_rag_disabled_passes_full_document(monkeypatch):
    """
    When RAG is globally disabled and the test case has use_rag=False,
    the full document text must be sent to the provider unchanged.
    """
    provider_mock = MagicMock()
    provider_mock.ask.return_value = "Answer from full doc."

    rag_off = RagConfig(enabled=False, top_k=3)
    service = _make_service_with_mocks(provider_mock, rag_config=rag_off)

    full_doc = "Full document text. " * 50  # long enough to be chunked

    test_case = PromptTestCase(
        id="t_no_rag",
        question="What is the policy?",
        expected_answer="",
        source_document="dummy.md",
        use_rag=False,
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_prompt_tests",
        lambda path: [test_case],
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_document",
        lambda path: full_doc,
    )
    service.engine.evaluate = MagicMock(return_value=[])

    results = service._run_suite("dummy_suite.yaml")

    assert len(results) == 1
    # The provider must receive the complete document, not a retrieved excerpt.
    call_args = provider_mock.ask.call_args
    assert call_args[0][1] == full_doc


def test_rag_per_case_flag_uses_retrieved_context(monkeypatch):
    """
    When use_rag=True on the test case (even with global enabled=False),
    the provider should receive only the retrieved chunk text, not the full doc.
    """
    provider_mock = MagicMock()
    provider_mock.ask.return_value = "Chunk-based answer."

    rag_off = RagConfig(enabled=False, top_k=2)
    service = _make_service_with_mocks(provider_mock, rag_config=rag_off)

    # Document with two distinct heading sections so chunker produces ≥2 chunks.
    doc_text = (
        "## Leave Policy\n"
        "Employees are entitled to 25 days of annual leave per year.\n\n"
        "## Sick Leave\n"
        "Employees must notify their manager as soon as possible.\n"
    )
    test_case = PromptTestCase(
        id="t_per_case_rag",
        question="How many days of annual leave are employees entitled to?",
        expected_answer="",
        source_document="dummy.md",
        use_rag=True,
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_prompt_tests",
        lambda path: [test_case],
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_document",
        lambda path: doc_text,
    )
    service.engine.evaluate = MagicMock(return_value=[])

    results = service._run_suite("dummy_suite.yaml")

    assert len(results) == 1
    call_args = provider_mock.ask.call_args
    context_sent = call_args[0][1]
    # Retrieval should have narrowed the context; it must NOT equal the full doc.
    assert context_sent != doc_text
    # The most relevant chunk should contain leave-days content.
    assert "25 days" in context_sent or "annual leave" in context_sent.lower()


def test_rag_global_flag_applies_to_all_cases(monkeypatch):
    """
    When RagConfig.enabled=True the retriever path is used even for a
    test case that has use_rag=False.
    """
    provider_mock = MagicMock()
    provider_mock.ask.return_value = "Global RAG answer."

    rag_on = RagConfig(enabled=True, top_k=1)
    service = _make_service_with_mocks(provider_mock, rag_config=rag_on)

    doc_text = (
        "## Policy A\nSome policy text here for section A.\n\n"
        "## Policy B\nDifferent policy text here for section B.\n"
    )
    test_case = PromptTestCase(
        id="t_global_rag",
        question="What is policy A?",
        expected_answer="",
        source_document="dummy.md",
        use_rag=False,  # global flag overrides this
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_prompt_tests",
        lambda path: [test_case],
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_document",
        lambda path: doc_text,
    )
    service.engine.evaluate = MagicMock(return_value=[])

    results = service._run_suite("dummy_suite.yaml")

    assert len(results) == 1
    context_sent = provider_mock.ask.call_args[0][1]
    # top_k=1 so we get exactly one chunk — definitely shorter than full doc.
    assert len(context_sent) < len(doc_text)


def test_rag_empty_document_falls_back_to_full_document(monkeypatch):
    """
    If chunking returns no chunks (empty document), ExecutionService must
    fall back to using the full (empty) document text rather than crashing.
    """
    provider_mock = MagicMock()
    provider_mock.ask.return_value = "Empty doc answer."

    rag_on = RagConfig(enabled=True, top_k=3)
    service = _make_service_with_mocks(provider_mock, rag_config=rag_on)

    test_case = PromptTestCase(
        id="t_empty",
        question="Any question?",
        expected_answer="",
        source_document="dummy.md",
        use_rag=True,
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_prompt_tests",
        lambda path: [test_case],
    )
    monkeypatch.setattr(
        "ai_orchestrator.services.execution_service.load_document",
        lambda path: "",  # empty document
    )
    service.engine.evaluate = MagicMock(return_value=[])

    # Must not raise; answer should be recorded.
    results = service._run_suite("dummy_suite.yaml")
    assert len(results) == 1
    assert results[0].error is None


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
