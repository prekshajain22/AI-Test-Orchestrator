import logging

from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.reporting.report_manager import ReportManager
from ai_orchestrator.services.execution_service import ExecutionService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)


class TestRunner:
    """
    Orchestrates a test run.

    Delegates execution to ExecutionService (reads config/test_suite.yaml
    and config/evaluation.yaml) and reporting to ReportManager.
    The runner only knows the sequence — not the details.
    """

    def __init__(self):
        self.execution_service = ExecutionService()
        self.report_manager = ReportManager()

    def run(self) -> list[TestExecutionResult]:
        results = self.execution_service.execute()
        self.report_manager.generate(results)
        return results
