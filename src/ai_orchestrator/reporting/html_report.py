from datetime import datetime
from pathlib import Path

from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.models.execution_summary import ExecutionSummary


_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px;
    color: #1a1a2e;
    background: #f4f6f9;
    padding: 32px;
}

h1 { font-size: 22px; font-weight: 700; }
h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; }
h3 { font-size: 14px; font-weight: 600; }

.header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 20px;
}
.header-meta { font-size: 12px; color: #666; margin-top: 4px; }

.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 4px;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.badge-pass { background: #d1fae5; color: #065f46; }
.badge-fail { background: #fee2e2; color: #991b1b; }

/* Metadata strip */
.meta-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-bottom: 24px;
}
.meta-chip {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 12px;
    color: #475569;
}
.meta-chip strong { color: #1e293b; }

/* Summary cards */
.cards {
    display: flex;
    gap: 16px;
    margin-bottom: 28px;
    flex-wrap: wrap;
}
.card {
    background: #fff;
    border-radius: 8px;
    padding: 20px 28px;
    flex: 1;
    min-width: 130px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
    text-align: center;
}
.card .value {
    font-size: 32px;
    font-weight: 700;
    line-height: 1.1;
}
.card .label { font-size: 12px; color: #666; margin-top: 4px; }
.card-pass .value  { color: #059669; }
.card-fail .value  { color: #dc2626; }
.card-total .value { color: #2563eb; }
.card-rate .value  { color: #7c3aed; }

.panel {
    background: #fff;
    border-radius: 8px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,.08);
}

/* Pass/Fail chart */
.chart-row {
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
}
.donut {
    width: 140px;
    height: 140px;
    border-radius: 50%;
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
}
.donut-hole {
    width: 90px;
    height: 90px;
    border-radius: 50%;
    background: #fff;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
.donut-hole .pct { font-size: 20px; font-weight: 700; color: #1e293b; }
.donut-hole .pct-label { font-size: 10px; color: #94a3b8; text-transform: uppercase; }

.legend { display: flex; flex-direction: column; gap: 10px; }
.legend-item { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #334155; }
.legend-dot { width: 12px; height: 12px; border-radius: 3px; flex-shrink: 0; }
.legend-dot.pass { background: #059669; }
.legend-dot.fail { background: #dc2626; }

.bars { flex: 1; display: flex; flex-direction: column; gap: 14px; min-width: 220px; }
.bar-row { display: flex; align-items: center; gap: 10px; }
.bar-label { width: 100px; font-size: 12px; color: #475569; font-weight: 600; }
.bar-track {
    flex: 1;
    height: 16px;
    border-radius: 4px;
    background: #f1f5f9;
    overflow: hidden;
    display: flex;
}
.bar-fill-pass { background: #059669; height: 100%; }
.bar-fill-fail { background: #dc2626; height: 100%; }
.bar-count { font-size: 12px; color: #475569; width: 40px; text-align: right; }

table { width: 100%; border-collapse: collapse; }
th {
    background: #f8fafc;
    text-align: left;
    padding: 10px 14px;
    font-size: 12px;
    font-weight: 600;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    border-bottom: 2px solid #e2e8f0;
}
td {
    padding: 10px 14px;
    border-bottom: 1px solid #f1f5f9;
    vertical-align: top;
}
tr:last-child td { border-bottom: none; }

.score-bar-wrap { display: flex; align-items: center; gap: 8px; }
.score-bar {
    height: 8px;
    border-radius: 4px;
    background: #e2e8f0;
    flex: 1;
    overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 4px; }
.score-value { font-size: 12px; font-weight: 600; width: 36px; text-align: right; }

.pill {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 99px;
    font-size: 11px;
    font-weight: 600;
}
.pill-pass { background: #d1fae5; color: #065f46; }
.pill-fail { background: #fee2e2; color: #991b1b; }

.test-card {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    margin-bottom: 16px;
    overflow: hidden;
}
.test-card:last-child { margin-bottom: 0; }
.test-card.is-failed { border-color: #fecaca; }

.test-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 18px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
}
.test-card.is-failed .test-header { background: #fef2f2; }
.test-id { font-weight: 700; font-size: 13px; color: #1e293b; }
.test-body { padding: 16px 18px; }

.qa-label {
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #94a3b8;
    margin-bottom: 4px;
}
.qa-text {
    margin-bottom: 14px;
    line-height: 1.6;
    color: #334155;
    white-space: pre-wrap;
}
.qa-text:last-of-type { margin-bottom: 0; }

.empty-state {
    text-align: center;
    padding: 24px;
    color: #94a3b8;
    font-size: 13px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 32px;
}
"""


def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#059669"
    if score >= 0.5:
        return "#d97706"
    return "#dc2626"


def _score_bar(score: float) -> str:
    pct = int(score * 100)
    color = _score_color(score)
    return (
        f'<div class="score-bar-wrap">'
        f'<div class="score-bar">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f"</div>"
        f'<span class="score-value" style="color:{color};">{score:.2f}</span>'
        f"</div>"
    )


def _pill(passed: bool) -> str:
    css = "pill-pass" if passed else "pill-fail"
    label = "PASS" if passed else "FAIL"
    return f'<span class="pill {css}">{label}</span>'


def _render_meta_strip(summary: ExecutionSummary) -> str:
    meta = summary.metadata
    if not meta:
        return ""

    suites = ", ".join(meta.test_suite) if meta.test_suite else "\u2014"

    chips = [
        ("Execution ID", meta.execution_id),
        ("Timestamp", meta.timestamp),
        ("Provider", meta.provider),
        ("Model", meta.model),
        ("Temperature", str(meta.temperature)),
        ("Test Suite", suites),
    ]

    items = "".join(
        f'<div class="meta-chip">{label}: <strong>{value}</strong></div>'
        for label, value in chips
    )
    return f'<div class="meta-strip">{items}</div>'


def _render_donut(pass_rate: float) -> str:
    pct = max(0.0, min(100.0, pass_rate))
    color = _score_color(pct / 100)
    gradient = f"conic-gradient({color} {pct}%, #e2e8f0 0)"
    return f"""
    <div class="donut" style="background:{gradient};">
        <div class="donut-hole">
            <span class="pct">{pct:.0f}%</span>
            <span class="pct-label">Pass Rate</span>
        </div>
    </div>
    """


def _render_metric_bars(summary: ExecutionSummary) -> str:
    if not summary.metric_stats:
        return '<div class="empty-state">No metrics recorded.</div>'

    rows = ""
    for metric, stats in sorted(summary.metric_stats.items()):
        total = stats.passed + stats.failed
        pass_pct = (stats.passed / total * 100) if total else 0
        fail_pct = 100 - pass_pct
        rows += f"""
        <div class="bar-row">
            <div class="bar-label">{metric}</div>
            <div class="bar-track">
                <div class="bar-fill-pass" style="width:{pass_pct:.1f}%;"></div>
                <div class="bar-fill-fail" style="width:{fail_pct:.1f}%;"></div>
            </div>
            <div class="bar-count">{stats.passed}/{total}</div>
        </div>
        """
    return f'<div class="bars">{rows}</div>'


def _render_chart_section(summary: ExecutionSummary) -> str:
    legend = f"""
    <div class="legend">
        <div class="legend-item"><span class="legend-dot pass"></span> Passed ({summary.passed})</div>
        <div class="legend-item"><span class="legend-dot fail"></span> Failed ({summary.failed})</div>
    </div>
    """
    donut = _render_donut(summary.pass_rate)
    bars = _render_metric_bars(summary)

    return f"""
    <div class="panel">
        <h2>Pass / Fail Overview</h2>
        <div class="chart-row">
            {donut}
            {legend}
            {bars}
        </div>
    </div>
    """


def _render_metric_table(summary: ExecutionSummary) -> str:
    if not summary.metric_stats:
        return '<div class="empty-state">No metrics recorded.</div>'

    rows = ""
    for metric, stats in sorted(summary.metric_stats.items()):
        rows += (
            "<tr>"
            f"<td><strong>{metric}</strong></td>"
            f"<td>{_score_bar(stats.average_score)}</td>"
            f"<td>{stats.min_score:.2f}</td>"
            f"<td>{stats.max_score:.2f}</td>"
            f"<td>{_pill(True)} {stats.passed}</td>"
            f"<td>{_pill(False)} {stats.failed}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Metric</th><th>Avg Score</th><th>Min</th>"
        "<th>Max</th><th>Passed</th><th>Failed</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>"
    )


def _render_test_card(result: TestExecutionResult) -> str:
    overall_passed = result.passed
    card_cls = "test-card" if overall_passed else "test-card is-failed"

    if result.error:
        eval_table = (
            f'<div class="empty-state">Provider error: {result.error}</div>'
        )
    else:
        eval_rows = ""
        for e in result.evaluations:
            eval_rows += (
                "<tr>"
                f"<td>{e.metric}</td>"
                f"<td>{_score_bar(e.score)}</td>"
                f"<td>{_pill(e.passed)}</td>"
                f"<td>{e.reason}</td>"
                "</tr>"
            )

        eval_table = (
            "<table><thead><tr>"
            "<th>Metric</th><th>Score</th><th>Status</th><th>Reason</th>"
            f"</tr></thead><tbody>{eval_rows}</tbody></table>"
        )

    answer_text = result.answer if not result.error else "(no answer — provider error)"

    return f"""
    <div class="{card_cls}">
        <div class="test-header">
            <span class="test-id">{result.test_id}</span>
            {_pill(overall_passed)}
        </div>
        <div class="test-body">
            <div class="qa-label">Question</div>
            <div class="qa-text">{result.question}</div>
            <div class="qa-label">Answer</div>
            <div class="qa-text">{answer_text}</div>
            <div class="qa-label" style="margin-top:14px;">Evaluations</div>
            {eval_table}
        </div>
    </div>
    """


def _render_failed_tests(results: list[TestExecutionResult]) -> str:
    failed_results = [r for r in results if not r.passed]

    if not failed_results:
        return '<div class="empty-state">No failed tests. All tests passed.</div>'

    return "".join(_render_test_card(r) for r in failed_results)


def _render(summary: ExecutionSummary, results: list[TestExecutionResult]) -> str:
    badge_cls = "badge-pass" if summary.overall_status == "PASSED" else "badge-fail"

    meta_strip = _render_meta_strip(summary)
    chart_section = _render_chart_section(summary)
    metric_section = _render_metric_table(summary)
    failed_section = _render_failed_tests(results)
    all_test_cards = "".join(_render_test_card(r) for r in results)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>AI Test Execution Report</title>
    <style>{_CSS}</style>
</head>
<body>

    <div class="header">
        <div>
            <h1>AI Test Execution Report</h1>
            <div class="header-meta">Generated: {summary.generated_at}</div>
        </div>
        <span class="badge {badge_cls}">{summary.overall_status}</span>
    </div>

    {meta_strip}

    <div class="cards">
        <div class="card card-total">
            <div class="value">{summary.total_tests}</div>
            <div class="label">Total Tests</div>
        </div>
        <div class="card card-pass">
            <div class="value">{summary.passed}</div>
            <div class="label">Passed</div>
        </div>
        <div class="card card-fail">
            <div class="value">{summary.failed}</div>
            <div class="label">Failed</div>
        </div>
        <div class="card card-rate">
            <div class="value">{summary.pass_rate}%</div>
            <div class="label">Pass Rate</div>
        </div>
    </div>

    {chart_section}

    <div class="panel">
        <h2>Evaluator Score Table</h2>
        {metric_section}
    </div>

    <div class="panel">
        <h2>Failed Tests</h2>
        {failed_section}
    </div>

    <div class="panel">
        <h2>All Test Results</h2>
        {all_test_cards}
    </div>

    <div class="footer">
        AI Test Orchestrator &mdash; {datetime.now().year}
    </div>

</body>
</html>"""


class HtmlReport:
    """
    Generates an HTML test execution report from an ExecutionSummary and results.
    """

    def __init__(self, output_dir: str = "reports/html"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def render(
        self,
        summary: ExecutionSummary,
        execution_results: list[TestExecutionResult],
    ) -> str:
        """Return the report as an HTML string (used by PdfReport)."""
        return _render(summary, execution_results)

    def generate(
        self,
        summary: ExecutionSummary,
        execution_results: list[TestExecutionResult],
    ) -> Path:
        """Write the HTML report to disk and return the file path."""
        html = _render(summary, execution_results)

        filename = f"execution_{datetime.now():%Y%m%d_%H%M%S}.html"
        output_file = self.output_dir / filename

        with output_file.open("w", encoding="utf-8") as f:
            f.write(html)

        return output_file
