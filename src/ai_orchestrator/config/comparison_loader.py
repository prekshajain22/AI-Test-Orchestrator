from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ai_orchestrator.config.settings import settings


@dataclass
class ComparisonRunConfig:
    """
    A single named run configuration used inside a comparison run.

    Only the retriever strategy and top_k vary per run — everything else
    (provider, test suites, evaluators, reports) comes from ``settings``
    (``.env``) so it is configured in exactly one place.

      Retriever  |  Hallucination  |  Faithfulness  |  …
      ───────────┼─────────────────┼────────────────┼───
      TF-IDF     │       0.91      │      0.83      │
      BM25       │       0.96      │      0.92      │
    """

    name: str
    retriever: str = "tfidf"
    top_k: int = 3

    @property
    def provider(self) -> str:
        """Always delegates to settings — ``.env`` is the single source."""
        return settings.provider


@dataclass
class ComparisonConfig:
    """
    Full configuration for a comparison run.

    ``runs`` comes from ``config/comparison.yaml``.
    Everything else (test_suites, evaluators, reports) comes from
    ``settings`` (``.env``) so there is exactly one place to change them.
    """

    __test__ = False

    runs: list[ComparisonRunConfig]
    test_suites: list[str] = field(default_factory=lambda: list(settings.test_suites))
    evaluators: list[str] = field(default_factory=lambda: list(settings.evaluators))
    reports: list[str] = field(default_factory=lambda: list(settings.reports))


def load_comparison_config(
    path: str = "config/comparison.yaml",
) -> ComparisonConfig:
    """
    Load comparison run configurations from YAML.

    Only the ``runs`` list (name + retriever + top_k) is read from the
    YAML file.  Provider, test suites, evaluators and reports come from
    ``settings`` (``.env``).
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    runs = [
        ComparisonRunConfig(
            name=r["name"],
            retriever=r.get("retriever", "tfidf"),
            top_k=int(r.get("top_k", 3)),
        )
        for r in data.get("runs", [])
    ]

    return ComparisonConfig(runs=runs)
