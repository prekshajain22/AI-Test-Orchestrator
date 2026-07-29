from .base import BaseEvaluator
from .engine import EvaluationEngine
from .hallucination import HallucinationEvaluator
from .relevance import RelevanceEvaluator
from .faithfulness import FaithfulnessEvaluator


class EvaluationFactory:
    """
    Creates evaluator instances by name.
    Allows evaluators to be selected via config/evaluation.yaml.
    """

    _registry: dict[str, type[BaseEvaluator]] = {
        "hallucination": HallucinationEvaluator,
        "relevance": RelevanceEvaluator,
        "faithfulness": FaithfulnessEvaluator,
    }

    @classmethod
    def create(cls, name: str) -> BaseEvaluator:
        if name not in cls._registry:
            raise ValueError(
                f"Unknown evaluator: '{name}'. "
                f"Available: {list(cls._registry)}"
            )
        return cls._registry[name]()

    @classmethod
    def create_all(cls, names: list[str]) -> list[BaseEvaluator]:
        return [cls.create(name) for name in names]


__all__ = [
    "BaseEvaluator",
    "EvaluationEngine",
    "HallucinationEvaluator",
    "RelevanceEvaluator",
    "FaithfulnessEvaluator",
    "EvaluationFactory",
]
