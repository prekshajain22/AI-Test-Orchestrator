from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TestSuiteConfig:
    __test__ = False  # Prevent pytest from collecting this as a test class

    provider: str
    tests: list[str]


@dataclass
class EvaluationConfig:
    evaluators: list[str]


def load_test_suite(path: str = "config/test_suite.yaml") -> TestSuiteConfig:
    """Load provider name and list of test suite paths from YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return TestSuiteConfig(
        provider=data.get("provider", "gemini"),
        tests=data.get("tests", []),
    )


def load_evaluation_config(path: str = "config/evaluation.yaml") -> EvaluationConfig:
    """Load the list of evaluator names from YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return EvaluationConfig(
        evaluators=data.get("evaluators", []),
    )
