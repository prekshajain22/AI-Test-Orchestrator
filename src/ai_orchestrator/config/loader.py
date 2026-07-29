from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ExecutionConfig:
    """
    Combined execution configuration loaded from config/execution.yaml.
      - provider: which LLM provider to use
      - test_suites: any number of prompt test suite YAML files
      - evaluators: which evaluators to run against each response
      - reports: which report formats to generate
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    provider: str
    test_suites: list[str]
    evaluators: list[str]
    reports: list[str] = field(default_factory=list)


def load_execution_config(path: str = "config/execution.yaml") -> ExecutionConfig:
    """Load provider, test suites, evaluators, and report formats from YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    provider_data = data.get("provider", "gemini")
    if isinstance(provider_data, dict):
        provider = provider_data.get("name", "gemini")
    else:
        provider = provider_data

    return ExecutionConfig(
        provider=provider,
        test_suites=data.get("test_suites", []),
        evaluators=data.get("evaluators", []),
        reports=data.get("reports", []),
    )
