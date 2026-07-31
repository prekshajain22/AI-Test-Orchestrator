from dataclasses import dataclass, field

from ai_orchestrator.config.settings import settings


@dataclass
class RagConfig:
    """
    RAG (Retrieval-Augmented Generation) retrieval settings.

    These values are sourced from ``settings`` (i.e. ``.env``):
      - RAG_ENABLED   → enabled
      - RAG_TOP_K     → top_k
      - RAG_RETRIEVER → retriever
    """

    enabled: bool = False
    top_k: int = 3
    retriever: str = "tfidf"


@dataclass
class ExecutionConfig:
    """
    Combined execution configuration.

    All fields are populated from ``settings`` (i.e. ``.env``).
    No YAML file is needed — ``.env`` is the single source of truth.
    """

    __test__ = False  # Prevent pytest from collecting this as a test class

    provider: str
    test_suites: list[str]
    evaluators: list[str]
    reports: list[str] = field(default_factory=list)
    rag: RagConfig = field(default_factory=RagConfig)


def load_execution_config() -> ExecutionConfig:
    """
    Build an ExecutionConfig from environment / ``.env`` settings.

    Every value comes from the ``Settings`` singleton — ``.env`` is the
    single source of truth.  No YAML file is read.

    To change provider, test suites, evaluators, reports or RAG settings
    edit ``.env`` (or export the corresponding environment variable) and
    restart the process.
    """
    return ExecutionConfig(
        provider=settings.provider,
        test_suites=list(settings.test_suites),
        evaluators=list(settings.evaluators),
        reports=list(settings.reports),
        rag=RagConfig(
            enabled=settings.rag_enabled,
            top_k=settings.rag_top_k,
            retriever=settings.rag_retriever,
        ),
    )
