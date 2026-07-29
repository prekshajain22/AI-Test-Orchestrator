from dataclasses import dataclass

from ai_orchestrator.models import EvaluationResult


@dataclass
class TestExecutionResult:
    test_id: str
    question: str
    answer: str
    evaluations: list[EvaluationResult]