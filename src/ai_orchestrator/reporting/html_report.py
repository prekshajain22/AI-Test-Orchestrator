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
    margin-bottom: 28px;
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

.test-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 12px 18px;
    background: #f8fafc;
    border-bottom: 1px solid #e2e8f0;
}
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


def _render_metric_table(summary: ExecutionSummary) -> str:
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
    overall_passed = all(e.passed for e in result.evaluations)

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

    return f"""
    <div class="test-card">
        <div class="test-header">
            <span class="test-id">{result.test_id}</span>
            {_pill(overall_passed)}
        </div>
        <div class="test-body">
            <div class="qa-label">Question</div>
            <div class="qa-text">{result.question}</div>
            <div class="qa-label">Answer</div>
            <div class="qa-text">{result.answer}</div>
            <div class="qa-label" style="margin-top:14px;">Evaluations</div>
            {eval_table}
        </div>
    </div>
    """


def _render(summary: ExecutionSummary, results: list[TestExecutionResult]) -> str:
    badge_cls = "badge-pass" if summary.overall_status == "PASSED" else "badge-fail"

    test_cards = "".join(_render_test_card(r) for r in results)
    metric_section = _render_metric_table(summary) if summary.metric_stats else "<p>No metrics.</p>"

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

    <div class="panel">
        <h2>Metric Statistics</h2>
        {metric_section}
    </div>

    <div class="panel">
        <h2>Test Results</h2>
        {test_cards}
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
