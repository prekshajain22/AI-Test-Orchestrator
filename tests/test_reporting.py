import json

import pytest

from ai_orchestrator.models.evaluation import EvaluationResult
from ai_orchestrator.models.execution_result import TestExecutionResult
from ai_orchestrator.reporting.summary import ExecutionSummaryBuilder
from ai_orchestrator.reporting.json_report import JsonReport
from ai_orchestrator.reporting.html_report import HtmlReport


def make_results(scores: list[float]) -> list[TestExecutionResult]:
    return [
        TestExecutionResult(
            test_id=f"t{i}",
            question="Q",
            answer="A",
            evaluations=[
                EvaluationResult(
                    test_id=f"t{i}",
                    metric="hallucination",
                    score=score,
                    passed=score >= 0.7,
                    reason="test",
                )
            ],
        )
        for i, score in enumerate(scores, start=1)
    ]


# ── ExecutionSummaryBuilder ────────────────────────────────────

def test_summary_total_count():
    summary = ExecutionSummaryBuilder.build(make_results([1.0, 0.8, 0.4]))
    assert summary.total_tests == 3


def test_summary_passed_and_failed_counts():
    summary = ExecutionSummaryBuilder.build(make_results([1.0, 0.8, 0.4]))
    assert summary.passed == 2
    assert summary.failed == 1


def test_summary_pass_rate():
    summary = ExecutionSummaryBuilder.build(make_results([1.0, 0.0]))
    assert summary.pass_rate == 50.0


def test_summary_overall_status_passed():
    summary = ExecutionSummaryBuilder.build(make_results([1.0, 0.9]))
    assert summary.overall_status == "PASSED"


def test_summary_overall_status_failed():
    summary = ExecutionSummaryBuilder.build(make_results([1.0, 0.4]))
    assert summary.overall_status == "FAILED"


def test_summary_metric_stats_average():
    summary = ExecutionSummaryBuilder.build(make_results([1.0, 0.5]))
    assert summary.metric_stats["hallucination"].average_score == 0.75


def test_summary_metric_stats_min_max():
    summary = ExecutionSummaryBuilder.build(make_results([0.3, 0.9]))
    stats = summary.metric_stats["hallucination"]
    assert stats.min_score == 0.3
    assert stats.max_score == 0.9


# ── JsonReport ─────────────────────────────────────────────────

def test_json_report_creates_file(tmp_path):
    results = make_results([1.0])
    summary = ExecutionSummaryBuilder.build(results)
    path = JsonReport(output_dir=str(tmp_path)).generate(summary, results)
    assert path.exists()


def test_json_report_structure(tmp_path):
    results = make_results([1.0, 0.4])
    summary = ExecutionSummaryBuilder.build(results)
    path = JsonReport(output_dir=str(tmp_path)).generate(summary, results)
    data = json.loads(path.read_text())
    assert "summary" in data
    assert "execution_results" in data
    assert data["summary"]["total_tests"] == 2
    assert len(data["execution_results"]) == 2


# ── HtmlReport ─────────────────────────────────────────────────

def test_html_report_creates_file(tmp_path):
    results = make_results([1.0])
    summary = ExecutionSummaryBuilder.build(results)
    path = HtmlReport(output_dir=str(tmp_path)).generate(summary, results)
    assert path.exists()


def test_html_report_contains_test_id(tmp_path):
    results = make_results([1.0])
    summary = ExecutionSummaryBuilder.build(results)
    path = HtmlReport(output_dir=str(tmp_path)).generate(summary, results)
    assert "t1" in path.read_text(encoding="utf-8")


def test_html_report_contains_status_badge(tmp_path):
    results = make_results([1.0])
    summary = ExecutionSummaryBuilder.build(results)
    path = HtmlReport(output_dir=str(tmp_path)).generate(summary, results)
    html = path.read_text(encoding="utf-8")
    assert "PASSED" in html or "FAILED" in html


def test_html_render_returns_valid_document():
    results = make_results([0.9])
    summary = ExecutionSummaryBuilder.build(results)
    html = HtmlReport().render(summary, results)
    assert isinstance(html, str)
    assert "<!DOCTYPE html>" in html
