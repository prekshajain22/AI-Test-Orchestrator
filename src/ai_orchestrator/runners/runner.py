import logging
import uuid
from datetime import datetime

from ai_orchestrator.config.loader import load_execution_config
from ai_orchestrator.config.settings import settings
from ai_orchestrator.models import ExecutionMetadata, TestExecutionResult
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
        self.config = config

        self.execution_service = ExecutionService(
            provider_name=config.provider,
            test_suites=config.test_suites,
            evaluators=config.evaluators,
            rag_config=config.rag,
        )
        self.report_manager = ReportManager(reports=config.reports)

    def _build_metadata(self) -> ExecutionMetadata:
        return ExecutionMetadata(
            execution_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            provider=self.config.provider,
            model=self.execution_service.model_name,
            temperature=settings.temperature,
            test_suite=self.config.test_suites,
        )

    def run(self) -> list[TestExecutionResult]:
        metadata = self._build_metadata()
        results = self.execution_service.execute()
        self.report_manager.generate(results, metadata=metadata)
        return results
