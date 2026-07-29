from dataclasses import dataclass

from ai_orchestrator.models import EvaluationResult


@dataclass
class TestExecutionResult:
    __test__ = False  # Prevent pytest from collecting this as a test class

    test_id: str
    question: str
    answer: str
    evaluations: list[EvaluationResult]
