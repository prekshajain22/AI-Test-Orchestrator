"""
Tests for the Dashboard and its internal data helpers.

All tests are fully offline — JSON reports are written to tmp_path fixtures,
and the Dashboard is pointed at those directories rather than the real
reports/json/ folder.
"""
import json
import textwrap
from pathlib import Path

import pytest

from ai_orchestrator.reporting.dashboard import (
    Dashboard,
    _extract_run_row,
    _load_json_reports,
    _render,
    _score_bg,
    _score_chip,
    _kpi_row,
    _trend_table,
    _provider_comparison,
)


# ---------------------------------------------------------------------------
# Minimal report fixture builders
# ---------------------------------------------------------------------------

def _make_report(
    *,
    generated_at: str = "2026-07-30T10:00:00",
    provider: str = "gemini",
    model: str = "gemini-flash",
    total: int = 5,
    passed: int = 3,
    failed: int = 2,
    pass_rate: float = 60.0,
    metrics: dict[str, float] | None = None,
    execution_id: str = "abc12345-0000-0000-0000-000000000000",
) -> dict:
    """Build a minimal JSON report dict matching the real structure."""
    metric_stats = {}
    for name, avg in (metrics or {"hallucination": 0.8, "faithfulness": 0.7}).items():
        metric_stats[name] = {
            "average_score": avg,
            "min_score": avg - 0.1,
            "max_score": avg + 0.1,
            "passed": passed,
            "failed": failed,
        }
    return {
        "summary": {
            "generated_at": generated_at,
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": pass_rate,
            "overall_status": "PASSED" if passed == total else "FAILED",
            "metric_stats": metric_stats,
            "metadata": {
                "execution_id": execution_id,
                "timestamp": generated_at,
                "provider": provider,
                "model": model,
                "temperature": 0.0,
                "test_suite": ["suite.yaml"],
            },
        },
        "execution_results": [],
    }


def _write_report(tmp_path: Path, filename: str, report: dict) -> Path:
    p = tmp_path / filename
    p.write_text(json.dumps(report), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _score_bg
# ---------------------------------------------------------------------------

def test_score_bg_green_for_high():
    assert _score_bg(0.9) == "#059669"


def test_score_bg_amber_for_mid():
    assert _score_bg(0.6) == "#d97706"


def test_score_bg_red_for_low():
    assert _score_bg(0.2) == "#dc2626"


def test_score_bg_boundary_0_75_is_green():
    assert _score_bg(0.75) == "#059669"


# ---------------------------------------------------------------------------
# _score_chip
# ---------------------------------------------------------------------------

def test_score_chip_contains_score():
    html = _score_chip(0.85)
    assert "0.85" in html


def test_score_chip_contains_score_chip_class():
    html = _score_chip(0.5)
    assert "score-chip" in html


# ---------------------------------------------------------------------------
# _load_json_reports
# ---------------------------------------------------------------------------

def test_load_json_reports_returns_empty_for_missing_dir(tmp_path):
    reports = _load_json_reports(str(tmp_path / "nonexistent"))
    assert reports == []


def test_load_json_reports_returns_empty_for_empty_dir(tmp_path):
    reports = _load_json_reports(str(tmp_path))
    assert reports == []


def test_load_json_reports_loads_all_execution_files(tmp_path):
    _write_report(tmp_path, "execution_20260730_100000.json", _make_report())
    _write_report(tmp_path, "execution_20260730_110000.json", _make_report())
    reports = _load_json_reports(str(tmp_path))
    assert len(reports) == 2


def test_load_json_reports_ignores_non_execution_files(tmp_path):
    _write_report(tmp_path, "execution_20260730_100000.json", _make_report())
    (tmp_path / "other_file.json").write_text("{}", encoding="utf-8")
    reports = _load_json_reports(str(tmp_path))
    assert len(reports) == 1


def test_load_json_reports_ignores_malformed_json(tmp_path):
    _write_report(tmp_path, "execution_20260730_100000.json", _make_report())
    (tmp_path / "execution_bad.json").write_text("not json {{{", encoding="utf-8")
    reports = _load_json_reports(str(tmp_path))
    assert len(reports) == 1


def test_load_json_reports_sorted_by_generated_at(tmp_path):
    _write_report(tmp_path, "execution_20260730_120000.json",
                  _make_report(generated_at="2026-07-30T12:00:00"))
    _write_report(tmp_path, "execution_20260730_090000.json",
                  _make_report(generated_at="2026-07-30T09:00:00"))
    reports = _load_json_reports(str(tmp_path))
    timestamps = [r["summary"]["generated_at"] for r in reports]
    assert timestamps == sorted(timestamps)


# ---------------------------------------------------------------------------
# _extract_run_row
# ---------------------------------------------------------------------------

def test_extract_run_row_returns_none_for_empty_report():
    assert _extract_run_row({}) is None


def test_extract_run_row_extracts_provider():
    row = _extract_run_row(_make_report(provider="gemini"))
    assert row["provider"] == "gemini"


def test_extract_run_row_extracts_total_and_pass_rate():
    row = _extract_run_row(_make_report(total=5, pass_rate=60.0))
    assert row["total"] == 5
    assert row["pass_rate"] == 60.0


def test_extract_run_row_extracts_metrics():
    row = _extract_run_row(_make_report(metrics={"hallucination": 0.9, "relevance": 0.7}))
    assert "hallucination" in row["metrics"]
    assert "relevance" in row["metrics"]
    assert row["metrics"]["hallucination"] == pytest.approx(0.9)


def test_extract_run_row_truncates_execution_id():
    row = _extract_run_row(_make_report(execution_id="abcdef12-0000-0000-0000-000000000000"))
    assert row["execution_id"].startswith("abcdef12")
    assert "…" in row["execution_id"]


# ---------------------------------------------------------------------------
# _kpi_row
# ---------------------------------------------------------------------------

def test_kpi_row_shows_total_runs():
    rows = [
        {"total": 5, "pass_rate": 60.0, "metrics": {"h": 0.8}},
        {"total": 3, "pass_rate": 100.0, "metrics": {"h": 0.9}},
    ]
    html = _kpi_row(rows)
    assert "2" in html  # 2 runs


def test_kpi_row_shows_total_tests():
    rows = [
        {"total": 5, "pass_rate": 60.0, "metrics": {}},
        {"total": 3, "pass_rate": 100.0, "metrics": {}},
    ]
    html = _kpi_row(rows)
    assert "8" in html  # 5 + 3 tests


def test_kpi_row_empty_rows():
    html = _kpi_row([])
    assert "0" in html


# ---------------------------------------------------------------------------
# _trend_table
# ---------------------------------------------------------------------------

def test_trend_table_no_data_message():
    html = _trend_table([])
    assert "No execution reports found" in html


def test_trend_table_contains_provider():
    rows = [{
        "timestamp": "2026-07-30T10:00:00",
        "provider": "gemini",
        "model": "flash",
        "retriever": "tfidf",
        "total": 5,
        "pass_rate": 80.0,
        "metrics": {"hallucination": 0.8},
    }]
    html = _trend_table(rows)
    assert "gemini" in html


def test_trend_table_contains_metric_column_header():
    rows = [{
        "timestamp": "2026-07-30T10:00:00",
        "provider": "gemini",
        "model": "flash",
        "retriever": "tfidf",
        "total": 5,
        "pass_rate": 80.0,
        "metrics": {"faithfulness": 0.75},
    }]
    html = _trend_table(rows)
    assert "faithfulness" in html


def test_trend_table_contains_pass_rate():
    rows = [{
        "timestamp": "2026-07-30T10:00:00",
        "provider": "gemini",
        "model": "flash",
        "retriever": "tfidf",
        "total": 5,
        "pass_rate": 80.0,
        "metrics": {},
    }]
    html = _trend_table(rows)
    assert "80.0" in html


# ---------------------------------------------------------------------------
# _provider_comparison
# ---------------------------------------------------------------------------

def test_provider_comparison_no_data_message():
    html = _provider_comparison([])
    assert "No data" in html


def test_provider_comparison_lists_provider():
    rows = [
        {"provider": "gemini", "metrics": {"hallucination": 0.8}},
        {"provider": "gemini", "metrics": {"hallucination": 0.9}},
    ]
    html = _provider_comparison(rows)
    assert "gemini" in html


def test_provider_comparison_averages_scores():
    rows = [
        {"provider": "gemini", "metrics": {"hallucination": 0.6}},
        {"provider": "gemini", "metrics": {"hallucination": 1.0}},
    ]
    html = _provider_comparison(rows)
    # avg = 0.8 → should appear in rendered score chip
    assert "0.80" in html


def test_provider_comparison_multiple_providers():
    rows = [
        {"provider": "gemini", "metrics": {"hallucination": 0.8}},
        {"provider": "openai", "metrics": {"hallucination": 0.9}},
    ]
    html = _provider_comparison(rows)
    assert "gemini" in html
    assert "openai" in html


# ---------------------------------------------------------------------------
# _render
# ---------------------------------------------------------------------------

def test_render_is_valid_html():
    rows = [{
        "timestamp": "2026-07-30T10:00:00",
        "provider": "gemini",
        "model": "flash",
        "retriever": "tfidf",
        "total": 5,
        "passed": 3,
        "failed": 2,
        "pass_rate": 60.0,
        "metrics": {"hallucination": 0.8},
    }]
    html = _render(rows, "2026-07-30T10:00:00")
    assert "<!DOCTYPE html>" in html
    assert "</html>" in html


def test_render_shows_run_count_in_subtitle():
    rows = [
        {"timestamp": "", "provider": "gemini", "model": "m", "retriever": "t",
         "total": 5, "passed": 3, "failed": 2, "pass_rate": 60.0, "metrics": {}},
        {"timestamp": "", "provider": "gemini", "model": "m", "retriever": "t",
         "total": 5, "passed": 3, "failed": 2, "pass_rate": 60.0, "metrics": {}},
    ]
    html = _render(rows, "2026-07-30")
    assert "2 run(s)" in html


def test_render_empty_rows_still_valid_html():
    html = _render([], "2026-07-30")
    assert "<!DOCTYPE html>" in html


# ---------------------------------------------------------------------------
# Dashboard.generate (end-to-end with real files)
# ---------------------------------------------------------------------------

def test_dashboard_generate_creates_file(tmp_path):
    reports_dir = tmp_path / "json"
    reports_dir.mkdir()
    _write_report(reports_dir, "execution_001.json", _make_report())

    output = tmp_path / "dashboard.html"
    dashboard = Dashboard(reports_dir=str(reports_dir), output_path=str(output))
    result = dashboard.generate()

    assert result == output
    assert output.exists()


def test_dashboard_generate_includes_provider_in_output(tmp_path):
    reports_dir = tmp_path / "json"
    reports_dir.mkdir()
    _write_report(reports_dir, "execution_001.json", _make_report(provider="gemini"))

    output = tmp_path / "dashboard.html"
    Dashboard(reports_dir=str(reports_dir), output_path=str(output)).generate()
    html = output.read_text(encoding="utf-8")
    assert "gemini" in html


def test_dashboard_generate_no_reports_still_creates_file(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    output = tmp_path / "dashboard.html"
    Dashboard(reports_dir=str(empty_dir), output_path=str(output)).generate()
    assert output.exists()


def test_dashboard_generate_creates_output_parent_dirs(tmp_path):
    reports_dir = tmp_path / "json"
    reports_dir.mkdir()
    _write_report(reports_dir, "execution_001.json", _make_report())

    nested_output = tmp_path / "nested" / "deep" / "dashboard.html"
    Dashboard(reports_dir=str(reports_dir), output_path=str(nested_output)).generate()
    assert nested_output.exists()


def test_dashboard_aggregates_multiple_reports(tmp_path):
    reports_dir = tmp_path / "json"
    reports_dir.mkdir()
    _write_report(reports_dir, "execution_001.json", _make_report(total=5, pass_rate=60.0))
    _write_report(reports_dir, "execution_002.json", _make_report(total=5, pass_rate=80.0))

    output = tmp_path / "dashboard.html"
    Dashboard(reports_dir=str(reports_dir), output_path=str(output)).generate()
    html = output.read_text(encoding="utf-8")
    # KPI total tests = 10 (5 + 5)
    assert "10" in html
