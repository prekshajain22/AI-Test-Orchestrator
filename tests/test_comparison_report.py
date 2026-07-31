"""
Tests for ComparisonReport and its internal helpers.

All tests use in-memory RunResult / ComparisonRunConfig fixtures — no
real providers, no file system (except where tmp_path is used to verify
the generated file).
"""
import pytest

from ai_orchestrator.config.comparison_loader import ComparisonRunConfig
from ai_orchestrator.models.evaluation import EvaluationResult
from ai_orchestrator.models.execution_result import TestExecutionResult
from ai_orchestrator.reporting.comparison_report import (
    ComparisonReport,
    _compute_metric_averages,
    _compute_pass_rates,
    _render,
    _score_color,
    _score_cell,
)
from ai_orchestrator.runners.comparison_runner import RunResult


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _run_config(name: str, retriever: str = "tfidf") -> ComparisonRunConfig:
    return ComparisonRunConfig(name=name, retriever=retriever, top_k=3)


def _eval(test_id: str, metric: str, score: float) -> EvaluationResult:
    return EvaluationResult(
        test_id=test_id, metric=metric, score=score,
        passed=score >= 0.7, reason="test",
    )


def _result(test_id: str, scores: dict[str, float]) -> TestExecutionResult:
    return TestExecutionResult(
        test_id=test_id,
        question="Q?",
        answer="A.",
        evaluations=[_eval(test_id, metric, score) for metric, score in scores.items()],
    )


def _run_result(name: str, results: list[TestExecutionResult], retriever: str = "tfidf") -> RunResult:
    return RunResult(run_config=_run_config(name, retriever=retriever), results=results)


# ---------------------------------------------------------------------------
# _score_color
# ---------------------------------------------------------------------------

def test_score_color_green_for_high_score():
    assert _score_color(0.9) == "#059669"


def test_score_color_amber_for_mid_score():
    assert _score_color(0.6) == "#d97706"


def test_score_color_red_for_low_score():
    assert _score_color(0.3) == "#dc2626"


def test_score_color_boundary_0_75_is_green():
    assert _score_color(0.75) == "#059669"


def test_score_color_boundary_0_5_is_amber():
    assert _score_color(0.5) == "#d97706"


# ---------------------------------------------------------------------------
# _score_cell
# ---------------------------------------------------------------------------

def test_score_cell_contains_score_value():
    html = _score_cell(0.85, is_best=False)
    assert "0.85" in html


def test_score_cell_best_contains_best_badge():
    html = _score_cell(0.9, is_best=True)
    assert "BEST" in html


def test_score_cell_non_best_has_no_badge():
    html = _score_cell(0.7, is_best=False)
    assert "BEST" not in html


def test_score_cell_best_has_best_css_class():
    html = _score_cell(0.9, is_best=True)
    assert "best" in html


# ---------------------------------------------------------------------------
# _compute_metric_averages
# ---------------------------------------------------------------------------

def test_compute_metric_averages_single_run_single_metric():
    rr = _run_result("Run A", [
        _result("t1", {"hallucination": 0.8}),
        _result("t2", {"hallucination": 0.6}),
    ])
    avgs = _compute_metric_averages([rr])
    assert avgs["Run A"]["hallucination"] == pytest.approx(0.7, abs=1e-3)


def test_compute_metric_averages_multiple_runs():
    rr1 = _run_result("Run A", [_result("t1", {"hallucination": 1.0})])
    rr2 = _run_result("Run B", [_result("t1", {"hallucination": 0.5})])
    avgs = _compute_metric_averages([rr1, rr2])
    assert avgs["Run A"]["hallucination"] == pytest.approx(1.0)
    assert avgs["Run B"]["hallucination"] == pytest.approx(0.5)


def test_compute_metric_averages_multiple_metrics():
    rr = _run_result("Run A", [
        _result("t1", {"hallucination": 0.8, "relevance": 0.6}),
    ])
    avgs = _compute_metric_averages([rr])
    assert "hallucination" in avgs["Run A"]
    assert "relevance" in avgs["Run A"]


def test_compute_metric_averages_empty_results():
    rr = _run_result("Run A", [])
    avgs = _compute_metric_averages([rr])
    assert avgs["Run A"] == {}


# ---------------------------------------------------------------------------
# _compute_pass_rates
# ---------------------------------------------------------------------------

def test_compute_pass_rates_all_passed():
    rr = _run_result("Run A", [
        _result("t1", {"hallucination": 1.0}),
        _result("t2", {"hallucination": 0.9}),
    ])
    rates = _compute_pass_rates([rr])
    assert rates["Run A"] == 100.0


def test_compute_pass_rates_none_passed():
    rr = _run_result("Run A", [
        _result("t1", {"hallucination": 0.3}),
    ])
    rates = _compute_pass_rates([rr])
    assert rates["Run A"] == 0.0


def test_compute_pass_rates_empty_results():
    rr = _run_result("Run A", [])
    rates = _compute_pass_rates([rr])
    assert rates["Run A"] == 0.0


def test_compute_pass_rates_partial():
    rr = _run_result("Run A", [
        _result("t1", {"hallucination": 1.0}),
        _result("t2", {"hallucination": 0.3}),
    ])
    rates = _compute_pass_rates([rr])
    assert rates["Run A"] == 50.0


# ---------------------------------------------------------------------------
# _render (HTML output)
# ---------------------------------------------------------------------------

def _two_run_results() -> list[RunResult]:
    return [
        _run_result("TF-IDF", [
            _result("t1", {"hallucination": 0.9, "relevance": 0.8}),
            _result("t2", {"hallucination": 0.7, "relevance": 0.6}),
        ], retriever="tfidf"),
        _run_result("BM25", [
            _result("t1", {"hallucination": 0.95, "relevance": 0.85}),
            _result("t2", {"hallucination": 0.75, "relevance": 0.65}),
        ], retriever="bm25"),
    ]


def test_render_is_valid_html():
    html = _render(_two_run_results())
    assert "<!DOCTYPE html>" in html
    assert "</html>" in html


def test_render_contains_run_names():
    html = _render(_two_run_results())
    assert "TF-IDF" in html
    assert "BM25" in html


def test_render_contains_metric_names():
    html = _render(_two_run_results())
    assert "hallucination" in html
    assert "relevance" in html


def test_render_contains_best_badge():
    html = _render(_two_run_results())
    assert "BEST" in html


def test_render_empty_run_list():
    html = _render([])
    assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# ComparisonReport.generate (file output)
# ---------------------------------------------------------------------------

def test_comparison_report_creates_file(tmp_path):
    report = ComparisonReport(output_dir=str(tmp_path))
    path = report.generate(_two_run_results())
    assert path.exists()
    assert path.suffix == ".html"


def test_comparison_report_filename_starts_with_comparison(tmp_path):
    report = ComparisonReport(output_dir=str(tmp_path))
    path = report.generate(_two_run_results())
    assert path.name.startswith("comparison_")


def test_comparison_report_file_contains_run_names(tmp_path):
    report = ComparisonReport(output_dir=str(tmp_path))
    path = report.generate(_two_run_results())
    html = path.read_text(encoding="utf-8")
    assert "TF-IDF" in html
    assert "BM25" in html


def test_comparison_report_output_dir_created(tmp_path):
    nested_dir = tmp_path / "nested" / "reports"
    report = ComparisonReport(output_dir=str(nested_dir))
    report.generate(_two_run_results())
    assert nested_dir.exists()
