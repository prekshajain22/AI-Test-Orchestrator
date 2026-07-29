from datetime import datetime

from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.models.execution_summary import ExecutionSummary, MetricStats


class ExecutionSummaryBuilder:
    """
    Builds an ExecutionSummary from a list of TestExecutionResult objects.

    Aggregates overall pass/fail counts and computes per-metric statistics
    (average score, min, max, passed, failed) across the full test run.
    """

    @staticmethod
    def build(results: list[TestExecutionResult]) -> ExecutionSummary:
        total = len(results)

        passed = sum(
            1 for r in results if all(e.passed for e in r.evaluations)
        )
        failed = total - passed
        pass_rate = round((passed / total) * 100, 1) if total > 0 else 0.0

        # Group all EvaluationResult objects by metric name
        metrics: dict = {}
        for result in results:
            for evaluation in result.evaluations:
                metrics.setdefault(evaluation.metric, []).append(evaluation)

        metric_stats = {
            metric: MetricStats(
                average_score=round(
                    sum(e.score for e in evals) / len(evals), 3
                ),
                min_score=min(e.score for e in evals),
                max_score=max(e.score for e in evals),
                passed=sum(1 for e in evals if e.passed),
                failed=sum(1 for e in evals if not e.passed),
            )
            for metric, evals in metrics.items()
        }

        return ExecutionSummary(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            total_tests=total,
            passed=passed,
            failed=failed,
            pass_rate=pass_rate,
            metric_stats=metric_stats,
        )
