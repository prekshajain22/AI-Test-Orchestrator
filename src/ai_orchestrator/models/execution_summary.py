from dataclasses import dataclass, field


@dataclass
class MetricStats:
    """
    Aggregated statistics for a single evaluation metric across all tests.
    """

    average_score: float
    min_score: float
    max_score: float
    passed: int
    failed: int


@dataclass
class ExecutionSummary:
    """
    Aggregated summary of a full test execution run.

    Contains overall pass/fail counts and per-metric statistics,
    giving a high-level view before drilling into individual results.
    """

    generated_at: str
    total_tests: int
    passed: int
    failed: int
    pass_rate: float  # 0.0 – 100.0
    metric_stats: dict[str, MetricStats] = field(default_factory=dict)

    @property
    def overall_status(self) -> str:
        return "PASSED" if self.failed == 0 else "FAILED"
