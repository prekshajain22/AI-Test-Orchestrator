from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ai_orchestrator.runners.comparison_runner import RunResult

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px; color: #1a1a2e; background: #f4f6f9; padding: 32px;
}
h1 { font-size: 22px; font-weight: 700; margin-bottom: 6px; }
h2 { font-size: 16px; font-weight: 600; margin-bottom: 14px; }
.subtitle { font-size: 12px; color: #666; margin-bottom: 28px; }
.panel {
    background: #fff; border-radius: 8px; padding: 24px;
    margin-bottom: 28px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
table { width: 100%; border-collapse: collapse; }
th {
    background: #f8fafc; text-align: left; padding: 10px 14px;
    font-size: 12px; font-weight: 600; color: #475569;
    text-transform: uppercase; letter-spacing: 0.4px;
    border-bottom: 2px solid #e2e8f0;
}
th.run-col { background: #1e293b; color: #f8fafc; text-align: center; }
td { padding: 10px 14px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
tr:last-child td { border-bottom: none; }
td.metric-name { font-weight: 600; color: #334155; }
td.score-cell { text-align: center; }
.score-wrap { display: flex; flex-direction: column; align-items: center; gap: 4px; }
.score-num { font-size: 18px; font-weight: 700; }
.score-bar { width: 80px; height: 6px; border-radius: 3px; background: #e2e8f0; overflow: hidden; }
.score-bar-fill { height: 100%; border-radius: 3px; }
.pill {
    display: inline-block; padding: 2px 8px; border-radius: 99px;
    font-size: 11px; font-weight: 600;
}
.pill-pass { background: #d1fae5; color: #065f46; }
.pill-fail { background: #fee2e2; color: #991b1b; }
.best { background: #fffbeb; }
.winner-badge {
    display: inline-block; margin-left: 6px;
    font-size: 10px; color: #92400e;
    font-weight: 700; letter-spacing: 0.5px;
}
.summary-cards { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 24px; }
.card {
    background: #fff; border-radius: 8px; padding: 18px 22px;
    flex: 1; min-width: 150px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
}
.card .run-name { font-size: 12px; color: #666; margin-bottom: 4px; }
.card .pass-rate { font-size: 28px; font-weight: 700; }
.card .label { font-size: 11px; color: #94a3b8; margin-top: 2px; }
.footer { text-align: center; color: #94a3b8; font-size: 12px; margin-top: 28px; }
"""


def _score_color(score: float) -> str:
    if score >= 0.75:
        return "#059669"
    if score >= 0.5:
        return "#d97706"
    return "#dc2626"


def _score_cell(score: float, is_best: bool) -> str:
    color = _score_color(score)
    pct = int(score * 100)
    badge = '<span class="winner-badge">★ BEST</span>' if is_best else ""
    return (
        f'<td class="score-cell{"  best" if is_best else ""}">'
        f'<div class="score-wrap">'
        f'<span class="score-num" style="color:{color};">{score:.2f}{badge}</span>'
        f'<div class="score-bar">'
        f'<div class="score-bar-fill" style="width:{pct}%;background:{color};"></div>'
        f"</div>"
        f"</div>"
        f"</td>"
    )


def _compute_metric_averages(
    run_results: list[RunResult],
) -> dict[str, dict[str, float]]:
    """
    Returns {run_name: {metric: avg_score}}.
    """
    averages: dict[str, dict[str, float]] = {}
    for rr in run_results:
        metric_scores: dict[str, list[float]] = {}
        for result in rr.results:
            for ev in result.evaluations:
                metric_scores.setdefault(ev.metric, []).append(ev.score)
        averages[rr.run_config.name] = {
            m: round(sum(v) / len(v), 3)
            for m, v in metric_scores.items()
        }
    return averages


def _compute_pass_rates(run_results: list[RunResult]) -> dict[str, float]:
    rates: dict[str, float] = {}
    for rr in run_results:
        total = len(rr.results)
        passed = sum(1 for r in rr.results if r.passed)
        rates[rr.run_config.name] = round((passed / total * 100) if total else 0.0, 1)
    return rates


def _render_summary_cards(
    run_results: list[RunResult],
    pass_rates: dict[str, float],
) -> str:
    cards = ""
    for rr in run_results:
        rate = pass_rates[rr.run_config.name]
        color = _score_color(rate / 100)
        cards += (
            f'<div class="card">'
            f'<div class="run-name">{rr.run_config.name}</div>'
            f'<div class="pass-rate" style="color:{color};">{rate}%</div>'
            f'<div class="label">Pass Rate &nbsp;|&nbsp; '
            f'{rr.run_config.provider} + {rr.run_config.retriever}</div>'
            f"</div>"
        )
    return f'<div class="summary-cards">{cards}</div>'


def _render_comparison_table(
    run_results: list[RunResult],
    averages: dict[str, dict[str, float]],
) -> str:
    # Collect all metrics across all runs
    all_metrics: set[str] = set()
    for run_avgs in averages.values():
        all_metrics.update(run_avgs.keys())
    all_metrics_sorted = sorted(all_metrics)

    run_names = [rr.run_config.name for rr in run_results]

    # Header row
    header_cols = "".join(
        f'<th class="run-col">{name}</th>' for name in run_names
    )
    header = f"<thead><tr><th>Metric</th>{header_cols}</tr></thead>"

    # Data rows
    rows = ""
    for metric in all_metrics_sorted:
        scores = {
            name: averages[name].get(metric, 0.0)
            for name in run_names
        }
        best_score = max(scores.values())
        cells = "".join(
            _score_cell(scores[name], scores[name] == best_score)
            for name in run_names
        )
        rows += f'<tr><td class="metric-name">{metric}</td>{cells}</tr>'

    return f"<table>{header}<tbody>{rows}</tbody></table>"


def _render(run_results: list[RunResult]) -> str:
    averages = _compute_metric_averages(run_results)
    pass_rates = _compute_pass_rates(run_results)
    now = datetime.now()

    summary_cards = _render_summary_cards(run_results, pass_rates)
    table = _render_comparison_table(run_results, averages)

    run_meta_rows = ""
    for rr in run_results:
        run_meta_rows += (
            "<tr>"
            f"<td><strong>{rr.run_config.name}</strong></td>"
            f"<td>{rr.run_config.provider}</td>"
            f"<td>{rr.run_config.retriever}</td>"
            f"<td>{rr.run_config.top_k}</td>"
            f"<td>{len(rr.results)}</td>"
            f'<td><span class="pill {"pill-pass" if pass_rates[rr.run_config.name] >= 50 else "pill-fail"}">'
            f'{pass_rates[rr.run_config.name]}%</span></td>'
            "</tr>"
        )

    config_table = (
        "<table><thead><tr>"
        "<th>Run Name</th><th>Provider</th><th>Retriever</th>"
        "<th>Top-K</th><th>Tests</th><th>Pass Rate</th>"
        f"</tr></thead><tbody>{run_meta_rows}</tbody></table>"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <title>AI Retriever Comparison Report</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>Retriever Comparison Report</h1>
  <div class="subtitle">Generated: {now.isoformat(timespec="seconds")} &nbsp;|&nbsp; {len(run_results)} run(s)</div>

  {summary_cards}

  <div class="panel">
    <h2>Run Configurations</h2>
    {config_table}
  </div>

  <div class="panel">
    <h2>Average Metric Scores by Run</h2>
    <p style="font-size:12px;color:#666;margin-bottom:14px;">
      ★ BEST marks the highest-scoring run for each metric.
      Scores are averaged across all test cases in the suite.
    </p>
    {table}
  </div>

  <div class="footer">AI Test Orchestrator &mdash; {now.year}</div>
</body>
</html>"""


class ComparisonReport:
    """
    Generates an HTML side-by-side comparison report from multiple RunResults.

    The report shows:
      - Summary cards: pass rate per run configuration
      - Configuration table: provider, retriever, top-k per run
      - Metric comparison table: average score per metric per run,
        with the best-performing configuration highlighted

    Output: reports/comparison/comparison_YYYYMMDD_HHMMSS.html
    """

    def __init__(self, output_dir: str = "reports/comparison") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, run_results: list[RunResult]) -> Path:
        html = _render(run_results)
        filename = f"comparison_{datetime.now():%Y%m%d_%H%M%S}.html"
        output_file = self.output_dir / filename
        output_file.write_text(html, encoding="utf-8")
        return output_file
