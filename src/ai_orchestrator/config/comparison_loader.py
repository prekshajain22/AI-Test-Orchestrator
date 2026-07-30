from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ComparisonRunConfig:
    """
    A single named run configuration used inside a comparison run.

    Each run specifies which provider and retriever to pair, giving the
    framework one axis for the comparison table:

      Provider  |  Retriever  |  Hallucination  |  Faithfulness  |  …
      ──────────┼─────────────┼─────────────────┼────────────────┼───
      Gemini    │  TF-IDF     │       0.91       │      0.83      │
      Gemini    │  BM25       │       0.96       │      0.92      │
    """

    name: str
    provider: str
    retriever: str = "tfidf"
    top_k: int = 3


@dataclass
class ComparisonConfig:
    """
    Full configuration for a comparison run loaded from
    config/comparison.yaml.

    Attributes:
        runs:        List of (name, provider, retriever) configurations to
                     compare.
        test_suites: Prompt suite YAML files to run for every configuration.
        evaluators:  Evaluators to run for every configuration.
        reports:     Report formats to generate (json, html, comparison).
    """

    __test__ = False

    runs: list[ComparisonRunConfig]
    test_suites: list[str]
    evaluators: list[str]
    reports: list[str] = field(default_factory=list)


def load_comparison_config(
    path: str = "config/comparison.yaml",
) -> ComparisonConfig:
    """Load a comparison run configuration from YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    runs = [
        ComparisonRunConfig(
            name=r["name"],
            provider=r["provider"],
            retriever=r.get("retriever", "tfidf"),
            top_k=int(r.get("top_k", 3)),
        )
        for r in data.get("runs", [])
    ]

    return ComparisonConfig(
        runs=runs,
        test_suites=data.get("test_suites", []),
        evaluators=data.get("evaluators", []),
        reports=data.get("reports", ["json", "html", "comparison"]),
    )
