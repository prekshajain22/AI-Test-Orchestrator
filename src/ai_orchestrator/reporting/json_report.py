import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from ai_orchestrator.models import TestExecutionResult


class JsonReport:
    """
    Generates a JSON report containing the results of a test execution.
    """

    def __init__(self, output_dir: str = "reports/json"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        execution_results: list[TestExecutionResult],
    ) -> Path:
        """
        Save execution results as a JSON report.
        """

        passed = sum(
            all(result.passed for result in test.evaluations)
            for test in execution_results
        )

        failed = len(execution_results) - passed

        report = {
            "execution_time": datetime.now().isoformat(timespec="seconds"),
            "total_tests": len(execution_results),
            "passed": passed,
            "failed": failed,
            "tests": [asdict(test) for test in execution_results],
        }

        filename = (
            f"execution_{datetime.now():%Y%m%d_%H%M%S}.json"
        )

        output_file = self.output_dir / filename

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=4)

        return output_file