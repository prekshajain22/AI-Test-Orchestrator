from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime

from ai_orchestrator.config.comparison_loader import ComparisonConfig, ComparisonRunConfig
from ai_orchestrator.config.loader import RagConfig
from ai_orchestrator.config.settings import settings
from ai_orchestrator.models import ExecutionMetadata, TestExecutionResult
from ai_orchestrator.reporting.report_manager import ReportManager
from ai_orchestrator.reporting.comparison_report import ComparisonReport
from ai_orchestrator.services.execution_service import ExecutionService

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    """
    The outcome of one named run configuration in a comparison run.

    Attributes:
        run_config:  The configuration that produced these results.
        results:     Per-test execution results.
        metadata:    Execution metadata (id, timestamp, provider, model).
    """

    run_config: ComparisonRunConfig
    results: list[TestExecutionResult] = field(default_factory=list)
    metadata: ExecutionMetadata | None = None


class ComparisonRunner:
    """
    Executes multiple (provider, retriever) configurations against the
    same test suites and produces a side-by-side comparison report.

    Usage::

        from ai_orchestrator.config.comparison_loader import load_comparison_config
        from ai_orchestrator.runners.comparison_runner import ComparisonRunner

        config = load_comparison_config("config/comparison.yaml")
        runner = ComparisonRunner(config)
        run_results = runner.run()

    Or via the CLI script::

        python scripts/compare_runs.py
    """

    def __init__(self, config: ComparisonConfig) -> None:
        self.config = config

    def run(self) -> list[RunResult]:
        """Execute all configured runs and generate reports."""
        run_results: list[RunResult] = []

        for run_cfg in self.config.runs:
            logger.info(
                "═" * 60 + "\nComparison run: %s  (provider=%s  retriever=%s)",
                run_cfg.name,
                run_cfg.provider,
                run_cfg.retriever,
            )

            rag_config = RagConfig(
                enabled=True,
                top_k=run_cfg.top_k,
                retriever=run_cfg.retriever,
            )

            service = ExecutionService(
                provider_name=run_cfg.provider,
                test_suites=self.config.test_suites,
                evaluators=self.config.evaluators,
                rag_config=rag_config,
            )

            metadata = ExecutionMetadata(
                execution_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(timespec="seconds"),
                provider=run_cfg.provider,
                model=service.model_name,
                temperature=settings.temperature,
                test_suite=self.config.test_suites,
            )

            results = service.execute()

            # Per-run individual reports (JSON + HTML)
            report_manager = ReportManager(
                reports=[r for r in self.config.reports if r != "comparison"]
            )
            report_manager.generate(results, metadata=metadata)

            run_results.append(
                RunResult(
                    run_config=run_cfg,
                    results=results,
                    metadata=metadata,
                )
            )

        # Comparison report across all runs
        if "comparison" in self.config.reports:
            comparison_path = ComparisonReport().generate(run_results)
            print(f"Comparison report: {comparison_path}")

        return run_results
