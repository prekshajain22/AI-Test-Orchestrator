from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _load_json_reports(reports_dir: str = "reports/json") -> list[dict]:
    """Load and parse all execution JSON reports, sorted by timestamp."""
    path = Path(reports_dir)
    if not path.exists():
        return []

    reports = []
    for json_file in sorted(path.glob("execution_*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            data["_filename"] = json_file.name
            reports.append(data)
        except (json.JSONDecodeError, OSError):
            pass

    # Sort by summary.generated_at if available
    reports.sort(key=lambda r: r.get("summary", {}).get("generated_at", ""))
    return reports


def _extract_run_row(report: dict) -> dict | None:
    """Extract a flat summary row from a single report for the trend table."""
    summary = report.get("summary", {})
    if not summary:
        return None

    meta = summary.get("metadata") or {}
    metric_stats = summary.get("metric_stats", {})

    row = {
        "timestamp": summary.get("generated_at", ""),
        "execution_id": meta.get("execution_id", "")[:8] + "…",
        "provider": meta.get("provider", "—"),
        "model": meta.get("model", "—"),
        "retriever": "—",
        "total": summary.get("total_tests", 0),
        "passed": summary.get("passed", 0),
        "failed": summary.get("failed", 0),
        "pass_rate": summary.get("pass_rate", 0.0),
        "metrics": {},
    }

    for metric, stats in metric_stats.items():
        row["metrics"][metric] = round(stats.get("average_score", 0.0), 3)

    return row


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 14px; color: #1a1a2e; background: #f0f4f8; padding: 28px;
}
h1 { font-size: 24px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 4px; }
h2 { font-size: 16px; font-weight: 600; margin-bottom: 14px; }
.subtitle { font-size: 12px; color: #64748b; margin-bottom: 32px; }
.kpi-row { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }
.kpi {
    background: #fff; border-radius: 10px; padding: 20px 26px;
    flex: 1; min-width: 130px; box-shadow: 0 1px 4px rgba(0,0,0,.07);
    text-align: center;
}
.kpi .value { font-size: 34px; font-weight: 800; line-height: 1.1; }
.kpi .label { font-size: 11px; color: #94a3b8; text-transform: uppercase; margin-top: 4px; letter-spacing: 0.4px; }
.kpi-blue  .value { color: #2563eb; }
.kpi-green .value { color: #059669; }
.kpi-red   .value { color: #dc2626; }
.kpi-purple .value { color: #7c3aed; }
.panel {
    background: #fff; border-radius: 10px; padding: 24px;
    margin-bottom: 24px; box-shadow: 0 1px 4px rgba(0,0,0,.07);
    overflow-x: auto;
}
table { width: 100%; border-collapse: collapse; min-width: 600px; }
th {
    background: #f8fafc; text-align: left; padding: 9px 12px;
    font-size: 11px; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 2px solid #e2e8f0; white-space: nowrap;
}
td { padding: 9px 12px; border-bottom: 1px solid #f1f5f9; vertical-align: middle; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #fafafa; }
.pill {
    display: inline-block; padding: 2px 8px; border-radius: 99px;
    font-size: 11px; font-weight: 600;
}
.pill-pass { background: #d1fae5; color: #065f46; }
.pill-fail { background: #fee2e2; color: #991b1b; }
.score-chip {
    display: inline-block; padding: 3px 8px; border-radius: 5px;
    font-size: 12px; font-weight: 700; color: #fff; white-space: nowrap;
}
.trend-up { color: #059669; font-size: 11px; }
.trend-dn { color: #dc2626; font-size: 11px; }
.badge {
    display: inline-block; padding: 3px 10px; border-radius: 5px;
    font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
    background: #1e293b; color: #f8fafc;
}
.footer { text-align: center; color: #94a3b8; font-size: 12px; margin-top: 28px; }
.no-data { color: #94a3b8; font-size: 13px; padding: 16px 0; }
"""

# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _score_bg(score: float) -> str:
    if score >= 0.75:
        return "#059669"
    if score >= 0.5:
        return "#d97706"
    return "#dc2626"


def _score_chip(score: float) -> str:
    bg = _score_bg(score)
    return f'<span class="score-chip" style="background:{bg};">{score:.2f}</span>'


def _pill(passed: bool, label: str = "") -> str:
    cls = "pill-pass" if passed else "pill-fail"
    text = label or ("PASS" if passed else "FAIL")
    return f'<span class="pill {cls}">{text}</span>'


def _kpi_row(rows: list[dict]) -> str:
    total_runs = len(rows)
    total_tests = sum(r["total"] for r in rows)
    avg_pass = round(sum(r["pass_rate"] for r in rows) / max(total_runs, 1), 1)

    all_scores: list[float] = []
    for r in rows:
        all_scores.extend(r["metrics"].values())
    avg_score = round(sum(all_scores) / max(len(all_scores), 1), 3)

    return f"""
    <div class="kpi-row">
      <div class="kpi kpi-blue">
        <div class="value">{total_runs}</div>
        <div class="label">Total Runs</div>
      </div>
      <div class="kpi kpi-blue">
        <div class="value">{total_tests}</div>
        <div class="label">Total Tests</div>
      </div>
      <div class="kpi kpi-green">
        <div class="value">{avg_pass}%</div>
        <div class="label">Avg Pass Rate</div>
      </div>
      <div class="kpi kpi-purple">
        <div class="value">{avg_score}</div>
        <div class="label">Avg Metric Score</div>
      </div>
    </div>
    """


def _trend_table(rows: list[dict]) -> str:
    if not rows:
        return '<div class="no-data">No execution reports found.</div>'

    all_metrics: list[str] = sorted(
        {m for r in rows for m in r["metrics"]}
    )
    metric_headers = "".join(f"<th>{m}</th>" for m in all_metrics)

    table_rows = ""
    for i, row in enumerate(rows):
        rate_passed = row["pass_rate"] >= 50
        metric_cells = "".join(
            _score_chip(row["metrics"].get(m, 0.0)) if row["metrics"].get(m) is not None
            else "<td>—</td>"
            for m in all_metrics
        )
        # Wrap each score chip in a <td>
        metric_tds = "".join(
            f"<td>{_score_chip(row['metrics'].get(m, 0.0))}</td>"
            if row["metrics"].get(m) is not None else "<td>—</td>"
            for m in all_metrics
        )

        table_rows += (
            "<tr>"
            f"<td>{row['timestamp'][:19]}</td>"
            f"<td><span class='badge'>{row['provider']}</span></td>"
            f"<td>{row['model']}</td>"
            f"<td>{row['retriever']}</td>"
            f"<td>{row['total']}</td>"
            f"<td>{_pill(rate_passed, str(row['pass_rate']) + '%')}</td>"
            f"{metric_tds}"
            "</tr>"
        )

    return (
        f"<table><thead><tr>"
        f"<th>Timestamp</th><th>Provider</th><th>Model</th><th>Retriever</th>"
        f"<th>Tests</th><th>Pass Rate</th>"
        f"{metric_headers}"
        f"</tr></thead><tbody>{table_rows}</tbody></table>"
    )


def _provider_comparison(rows: list[dict]) -> str:
    """Group rows by provider and show average metric scores per provider."""
    if not rows:
        return '<div class="no-data">No data.</div>'

    provider_metrics: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        p = row["provider"]
        provider_metrics.setdefault(p, {})
        for m, v in row["metrics"].items():
            provider_metrics[p].setdefault(m, []).append(v)

    all_metrics = sorted({m for pm in provider_metrics.values() for m in pm})
    metric_headers = "".join(f"<th>{m}</th>" for m in all_metrics)

    table_rows = ""
    for provider, pm in sorted(provider_metrics.items()):
        avgs = {m: round(sum(pm.get(m, [0])) / max(len(pm.get(m, [0])), 1), 3)
                for m in all_metrics}
        metric_tds = "".join(f"<td>{_score_chip(avgs[m])}</td>" for m in all_metrics)
        table_rows += (
            f"<tr><td><strong>{provider}</strong></td>"
            f"<td>{len([r for r in rows if r['provider'] == provider])}</td>"
            f"{metric_tds}</tr>"
        )

    return (
        f"<table><thead><tr><th>Provider</th><th>Runs</th>"
        f"{metric_headers}</tr></thead><tbody>{table_rows}</tbody></table>"
    )


def _render(rows: list[dict], generated_at: str) -> str:
    kpis = _kpi_row(rows)
    trend = _trend_table(rows)
    prov_cmp = _provider_comparison(rows)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>AI Test Orchestrator — Dashboard</title>
  <style>{_CSS}</style>
</head>
<body>
  <h1>AI Test Orchestrator</h1>
  <div class="subtitle">Dashboard &nbsp;·&nbsp; Generated: {generated_at} &nbsp;·&nbsp; {len(rows)} run(s) loaded</div>

  {kpis}

  <div class="panel">
    <h2>Execution History</h2>
    {trend}
  </div>

  <div class="panel">
    <h2>Provider Comparison (averages across all runs)</h2>
    {prov_cmp}
  </div>

  <div class="footer">AI Test Orchestrator Dashboard &mdash; {datetime.now().year}</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Dashboard:
    """
    Aggregates all execution JSON reports in ``reports/json/`` and generates
    a single-page HTML dashboard showing:

    - KPI cards: total runs, total tests, average pass rate, average score
    - Execution history table with per-metric scores for every run
    - Provider comparison table (averages grouped by provider)

    Output: ``reports/dashboard.html``
    """

    def __init__(
        self,
        reports_dir: str = "reports/json",
        output_path: str = "reports/dashboard.html",
    ) -> None:
        self.reports_dir = reports_dir
        self.output_path = Path(output_path)

    def generate(self) -> Path:
        raw_reports = _load_json_reports(self.reports_dir)
        rows = [r for r in (_extract_run_row(rep) for rep in raw_reports) if r]

        generated_at = datetime.now().isoformat(timespec="seconds")
        html = _render(rows, generated_at)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")
        return self.output_path
