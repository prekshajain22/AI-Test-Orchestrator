from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RagConfig:
    """
    RAG (Retrieval-Augmented Generation) retrieval settings.

    Controls whether ExecutionService chunks source documents and uses
    a retriever to select only the most relevant passages as LLM context,
    rather than always passing the entire document.

    Attributes:
        enabled: Global on/off switch.  When False every test case uses the
            full document as context (legacy behaviour).  Individual
            PromptTestCases can still opt in by setting use_rag=True even
            when the global flag is False.
        top_k: How many chunks the retriever should return per question.
        retriever: Which retrieval strategy to use.  Currently only
            "tfidf" is supported.
    """

    enabled: bool = False
    top_k: int = 3
    retriever: str = "tfidf"


@dataclass
class ExecutionConfig:
    """
    Combined execution configuration loaded from config/execution.yaml.
      - provider: which LLM provider to use
      - test_suites: any number of prompt test suite YAML files
      - evaluators: which evaluators to run against each response
      - reports: which report formats to generate
      - rag: RAG retrieval settings (opt-in, disabled by default)
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    provider: str
    test_suites: list[str]
    evaluators: list[str]
    reports: list[str] = field(default_factory=list)
    rag: RagConfig = field(default_factory=RagConfig)


def load_execution_config(path: str = "config/execution.yaml") -> ExecutionConfig:
    """Load provider, test suites, evaluators, and report formats from YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    provider_data = data.get("provider", "gemini")
    if isinstance(provider_data, dict):
        provider = provider_data.get("name", "gemini")
    else:
        provider = provider_data

    rag_data = data.get("rag", {}) or {}
    rag_config = RagConfig(
        enabled=bool(rag_data.get("enabled", False)),
        top_k=int(rag_data.get("top_k", 3)),
        retriever=str(rag_data.get("retriever", "tfidf")),
    )

    return ExecutionConfig(
        provider=provider,
        test_suites=data.get("test_suites", []),
        evaluators=data.get("evaluators", []),
        reports=data.get("reports", []),
        rag=rag_config,
    )
