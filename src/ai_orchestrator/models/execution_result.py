from dataclasses import dataclass

from ai_orchestrator.models import EvaluationResult


@dataclass
class TestExecutionResult:
    __test__ = False  # Prevent pytest from collecting this as a test class

    test_id: str
    question: str
    answer: str
    evaluations: list[EvaluationResult]
    error: str | None = None
    """
    Set when the provider itself failed to produce an answer (e.g. rate
    limit / quota exceeded). When set, `answer` and `evaluations` should be
    ignored/empty — this test must NEVER be scored as a real response.
    """

    @property
    def passed(self) -> bool:
        """
        A test only counts as passed if the provider succeeded (no error)
        and every evaluator passed. Tests with a provider error, or with
        no evaluations at all, are never considered passed.
        """
        return (
            self.error is None
            and bool(self.evaluations)
            and all(e.passed for e in self.evaluations)
        )
