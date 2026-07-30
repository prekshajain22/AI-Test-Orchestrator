"""
Dashboard generator script.

Reads all execution JSON reports from reports/json/ and generates a
single-page HTML dashboard at reports/dashboard.html.

Usage:
    python scripts/generate_dashboard.py
"""
from ai_orchestrator.reporting.dashboard import Dashboard

dashboard = Dashboard()
path = dashboard.generate()
print(f"Dashboard: {path}")
