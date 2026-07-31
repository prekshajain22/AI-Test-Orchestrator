"""
Tests for ComparisonRunner.

All tests mock ExecutionService, ReportManager, and ComparisonReport so no
real providers, file I/O, or network calls are made.
"""
import textwrap
from unittest.mock import MagicMock, patch

import pytest

from ai_orchestrator.config.comparison_loader import (
    ComparisonConfig,
    ComparisonRunConfig,
    load_comparison_config,
)
from ai_orchestrator.models.evaluation import EvaluationResult
from ai_orchestrator.models.execution_result import TestExecutionResult
from ai_orchestrator.runners.comparison_runner import ComparisonRunner, RunResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_result(test_id: str = "t1", passed: bool = True) -> TestExecutionResult:
    score = 1.0 if passed else 0.3
    return TestExecutionResult(
        test_id=test_id,
        question="Q?",
        answer="A.",
        evaluations=[
            EvaluationResult(
                test_id=test_id, metric="hallucination",
                score=score, passed=passed, reason="ok",
            )
        ],
    )


def _make_config(
    run_names: list[str] | None = None,
    reports: list[str] | None = None,
) -> ComparisonConfig:
    run_names = run_names or ["Run A", "Run B"]
    return ComparisonConfig(
        runs=[
            ComparisonRunConfig(name=n, retriever="tfidf", top_k=3)
            for n in run_names
        ],
        test_suites=["suite.yaml"],
        evaluators=["hallucination"],
        reports=reports or ["json", "html", "comparison"],
    )


def _patch_service(fake_results: list[TestExecutionResult]):
    """Context manager that patches ExecutionService to return fake_results."""
    mock_service = MagicMock()
    mock_service.execute.return_value = fake_results
    mock_service.model_name = "test-model"
    return patch(
        "ai_orchestrator.runners.comparison_runner.ExecutionService",
        return_value=mock_service,
    )


# ---------------------------------------------------------------------------
# load_comparison_config
# ---------------------------------------------------------------------------

def test_load_comparison_config_parses_runs(tmp_path):
    yaml_text = textwrap.dedent("""\
        runs:
          - name: "TF-IDF"
            retriever: tfidf
            top_k: 3
          - name: "BM25"
            retriever: bm25
            top_k: 5
    """)
    p = tmp_path / "comparison.yaml"
    p.write_text(yaml_text)
    config = load_comparison_config(str(p))
    assert len(config.runs) == 2
    assert config.runs[0].name == "TF-IDF"
    assert config.runs[1].retriever == "bm25"
    assert config.runs[1].top_k == 5


def test_load_comparison_config_defaults_retriever_to_tfidf(tmp_path):
    yaml_text = textwrap.dedent("""\
        runs:
          - name: "Run A"
    """)
    p = tmp_path / "comparison.yaml"
    p.write_text(yaml_text)
    config = load_comparison_config(str(p))
    assert config.runs[0].retriever == "tfidf"
    assert config.runs[0].top_k == 3


def test_load_comparison_config_provider_comes_from_settings(tmp_path):
    """Provider always comes from settings (.env), not from the YAML file."""
    yaml_text = textwrap.dedent("""\
        runs:
          - name: "Run A"
            retriever: bm25
    """)
    p = tmp_path / "comparison.yaml"
    p.write_text(yaml_text)
    config = load_comparison_config(str(p))
    # provider property delegates to settings — should never be empty
    assert len(config.runs[0].provider) > 0


# ---------------------------------------------------------------------------
# ComparisonRunConfig
# ---------------------------------------------------------------------------

def test_comparison_run_config_defaults():
    rc = ComparisonRunConfig(name="Test")
    assert rc.retriever == "tfidf"
    assert rc.top_k == 3


def test_comparison_run_config_provider_is_from_settings():
    rc = ComparisonRunConfig(name="Test", retriever="bm25")
    # provider is a property that reads settings.provider
    from ai_orchestrator.config.settings import settings
    assert rc.provider == settings.provider


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------

def test_run_result_defaults_to_empty_results():
    rc = ComparisonRunConfig(name="Run A")
    rr = RunResult(run_config=rc)
    assert rr.results == []
    assert rr.metadata is None


# ---------------------------------------------------------------------------
# ComparisonRunner.run — number of runs
# ---------------------------------------------------------------------------

def test_comparison_runner_returns_one_result_per_run():
    config = _make_config(run_names=["Run A", "Run B", "Run C"])
    with _patch_service([_fake_result()]), \
         patch("ai_orchestrator.runners.comparison_runner.ReportManager"), \
         patch("ai_orchestrator.runners.comparison_runner.ComparisonReport") as MockReport:
        MockReport.return_value.generate.return_value = "reports/comparison/x.html"
        runner = ComparisonRunner(config)
        results = runner.run()
    assert len(results) == 3


def test_comparison_runner_results_have_correct_run_names():
    config = _make_config(run_names=["Alpha", "Beta"])
    with _patch_service([_fake_result()]), \
         patch("ai_orchestrator.runners.comparison_runner.ReportManager"), \
         patch("ai_orchestrator.runners.comparison_runner.ComparisonReport") as MockReport:
        MockReport.return_value.generate.return_value = "x.html"
        runner = ComparisonRunner(config)
        results = runner.run()
    names = [rr.run_config.name for rr in results]
    assert names == ["Alpha", "Beta"]


def test_comparison_runner_each_result_stores_execution_results():
    fake = [_fake_result("t1"), _fake_result("t2")]
    config = _make_config(run_names=["Run A"])
    with _patch_service(fake), \
         patch("ai_orchestrator.runners.comparison_runner.ReportManager"), \
         patch("ai_orchestrator.runners.comparison_runner.ComparisonReport") as MockReport:
        MockReport.return_value.generate.return_value = "x.html"
        runner = ComparisonRunner(config)
        results = runner.run()
    assert len(results[0].results) == 2


def test_comparison_runner_metadata_is_populated():
    config = _make_config(run_names=["Run A"])
    with _patch_service([_fake_result()]), \
         patch("ai_orchestrator.runners.comparison_runner.ReportManager"), \
         patch("ai_orchestrator.runners.comparison_runner.ComparisonReport") as MockReport:
        MockReport.return_value.generate.return_value = "x.html"
        runner = ComparisonRunner(config)
        results = runner.run()
    assert results[0].metadata is not None
    assert results[0].metadata.provider == "gemini"


# ---------------------------------------------------------------------------
# ComparisonRunner.run — comparison report generation
# ---------------------------------------------------------------------------

def test_comparison_runner_generates_comparison_report_when_in_reports():
    config = _make_config(reports=["json", "comparison"])
    with _patch_service([_fake_result()]), \
         patch("ai_orchestrator.runners.comparison_runner.ReportManager"), \
         patch("ai_orchestrator.runners.comparison_runner.ComparisonReport") as MockReport:
        mock_instance = MockReport.return_value
        mock_instance.generate.return_value = "reports/x.html"
        runner = ComparisonRunner(config)
        runner.run()
    mock_instance.generate.assert_called_once()


def test_comparison_runner_skips_comparison_report_when_not_in_reports():
    config = _make_config(reports=["json", "html"])
    with _patch_service([_fake_result()]), \
         patch("ai_orchestrator.runners.comparison_runner.ReportManager"), \
         patch("ai_orchestrator.runners.comparison_runner.ComparisonReport") as MockReport:
        mock_instance = MockReport.return_value
        runner = ComparisonRunner(config)
        runner.run()
    mock_instance.generate.assert_not_called()


# ---------------------------------------------------------------------------
# ComparisonRunner.run — RAG config passed correctly
# ---------------------------------------------------------------------------

def test_comparison_runner_passes_rag_config_with_retriever():
    config = ComparisonConfig(
        runs=[ComparisonRunConfig(name="BM25 Run", retriever="bm25", top_k=5)],
        test_suites=["suite.yaml"],
        evaluators=["hallucination"],
        reports=["json"],
    )
    with patch("ai_orchestrator.runners.comparison_runner.ExecutionService") as MockService, \
         patch("ai_orchestrator.runners.comparison_runner.ReportManager"):
        mock_instance = MockService.return_value
        mock_instance.execute.return_value = [_fake_result()]
        mock_instance.model_name = "test-model"
        runner = ComparisonRunner(config)
        runner.run()

    call_kwargs = MockService.call_args.kwargs
    assert call_kwargs["rag_config"].retriever == "bm25"
    assert call_kwargs["rag_config"].top_k == 5
    assert call_kwargs["rag_config"].enabled is True


# ---------------------------------------------------------------------------
# ComparisonRunner.run — empty runs list
# ---------------------------------------------------------------------------

def test_comparison_runner_empty_runs_returns_empty_list():
    config = ComparisonConfig(
        runs=[],
        reports=["comparison"],
    )
    with patch("ai_orchestrator.runners.comparison_runner.ComparisonReport") as MockReport:
        mock_instance = MockReport.return_value
        mock_instance.generate.return_value = "x.html"
        runner = ComparisonRunner(config)
        results = runner.run()
    assert results == []
