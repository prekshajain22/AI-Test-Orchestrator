from dataclasses import dataclass


@dataclass
class EvaluationResult:
    """
    Represents the result of an AI response evaluation.
    """

    test_id: str
    metric: str
    score: float
    passed: bool
    reason: str