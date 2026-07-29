import logging

from ai_orchestrator.config.loader import load_execution_config
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

    Loads config/execution.yaml (provider, test suites, evaluators, reports)
    and delegates execution to ExecutionService and reporting to
    ReportManager. The runner only knows the sequence — not the details.
    """

    EXECUTION_CONFIG = "config/execution.yaml"

    def __init__(self):
        config = load_execution_config(self.EXECUTION_CONFIG)

        self.execution_service = ExecutionService(
            provider_name=config.provider,
            test_suites=config.test_suites,
            evaluators=config.evaluators,
        )
        self.report_manager = ReportManager(reports=config.reports)

    def run(self) -> list[TestExecutionResult]:
        results = self.execution_service.execute()
        self.report_manager.generate(results)
        return results
