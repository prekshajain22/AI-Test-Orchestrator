import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.models.execution_summary import ExecutionSummary


class JsonReport:
    """
    Generates a JSON report containing an execution summary and full results.
    """

    def __init__(self, output_dir: str = "reports/json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        summary: ExecutionSummary,
        execution_results: list[TestExecutionResult],
    ) -> Path:
        """
        Save execution summary and raw results as a JSON report.

        The report contains two top-level sections:
          - summary: aggregated statistics (pass rate, per-metric scores, etc.)
          - execution_results: the full per-test detail
        """
        report = {
            "summary": asdict(summary),
            "execution_results": [asdict(r) for r in execution_results],
        }

        filename = f"execution_{datetime.now():%Y%m%d_%H%M%S}.json"
        output_file = self.output_dir / filename

        with output_file.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

        return output_file
