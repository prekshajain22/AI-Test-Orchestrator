import logging
from typing import Optional

from ai_orchestrator.config.loader import RagConfig
from ai_orchestrator.evaluators.engine import EvaluationEngine
from ai_orchestrator.evaluators import EvaluationFactory
from ai_orchestrator.loaders import load_document, load_prompt_tests
from ai_orchestrator.loaders.chunker import chunk_document
from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.providers import ProviderFactory
from ai_orchestrator.providers.base import ProviderError
from ai_orchestrator.retrievers import TfidfRetriever

logger = logging.getLogger(__name__)

# Registry of available retrievers by name — extend as new strategies land.
_RETRIEVER_REGISTRY = {
    "tfidf": TfidfRetriever,
}


def _build_retriever(retriever_name: str):
    """Instantiate a retriever by name, raising a clear error if unknown."""
    cls = _RETRIEVER_REGISTRY.get(retriever_name)
    if cls is None:
        available = list(_RETRIEVER_REGISTRY)
        raise ValueError(
            f"Unknown retriever: '{retriever_name}'. "
            f"Available: {available}"
        )
    return cls()


class ExecutionService:
    """
    Executes a full test run end-to-end.

    Receives everything it needs via its constructor instead of reading
    configuration files itself:
      - provider_name: which LLM provider to use (e.g. "gemini")
      - test_suites: any number of prompt test suite YAML file paths
      - evaluators: which evaluators to run against each AI response
      - rag_config: optional RAG retrieval settings (disabled by default)

    RAG behaviour (controlled by ``rag_config``):
      - When ``rag_config`` is None **or** ``rag_config.enabled`` is False,
        *and* the individual ``PromptTestCase.use_rag`` is also False, the
        service loads the full source document text and passes it to the LLM
        unchanged — preserving the original behaviour exactly.
      - A test case is run in RAG mode when **either** the global flag
        ``rag_config.enabled`` is True **or** the per-case flag
        ``PromptTestCase.use_rag`` is True.  In that case the document is
        split into chunks (``chunk_document``) and only the ``top_k`` most
        relevant chunks (ranked by the configured retriever) are joined and
        used as context.

    Configuration loading is the caller's responsibility (see
    ai_orchestrator.config.loader.load_execution_config and
    ai_orchestrator.runners.runner.TestRunner), which keeps this service
    decoupled from how/where config is stored, and combinable with any
    number of test suite files.

    Does not know about reporting or how results are displayed.
    """

    def __init__(
        self,
        provider_name: str,
        test_suites: list[str],
        evaluators: list[str],
        rag_config: Optional[RagConfig] = None,
    ):
        self.provider_name = provider_name
        self.provider = ProviderFactory.create(provider_name)
        self.test_paths = test_suites

        # RAG retrieval — build once, reuse for every test.
        self._rag_config: RagConfig = rag_config or RagConfig()
        self._retriever = _build_retriever(self._rag_config.retriever)

        self.engine = EvaluationEngine()
        for evaluator in EvaluationFactory.create_all(evaluators):
            self.engine.register(evaluator)

    @property
    def model_name(self) -> str:
        """Best-effort lookup of the model name used by the active provider."""
        return getattr(self.provider, "model", None) or getattr(
            self.provider, "model_name", "unknown"
        )

    def execute(self) -> list[TestExecutionResult]:
        """Run all configured test suites."""
        execution_results: list[TestExecutionResult] = []

        for prompts_path in self.test_paths:
            logger.info("Loading test suite: %s", prompts_path)
            execution_results.extend(self._run_suite(prompts_path))

        logger.info("Execution complete. %d test(s) run.", len(execution_results))
        return execution_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _use_rag_for(self, test) -> bool:
        """Return True when this test case should be run in RAG mode."""
        return self._rag_config.enabled or test.use_rag

    def _build_rag_context(self, document_text: str, question: str) -> str:
        """
        Chunk *document_text*, retrieve the top-k most relevant chunks for
        *question*, and return their text joined by double newlines.

        Returns the full document unchanged if chunking produces no chunks
        (e.g. empty document), so the LLM call is never left without context.
        """
        chunks = chunk_document(document_text)
        if not chunks:
            logger.warning("chunk_document returned no chunks — using full document.")
            return document_text

        top_k = self._rag_config.top_k
        retrieved = self._retriever.retrieve(question, chunks, top_k=top_k)

        if not retrieved:
            logger.warning(
                "Retriever returned no results for question %r — "
                "falling back to full document.",
                question,
            )
            return document_text

        logger.debug(
            "RAG: retrieved %d/%d chunks (top_k=%d) for question %r",
            len(retrieved),
            len(chunks),
            top_k,
            question,
        )

        return "\n\n".join(rc.chunk.text for rc in retrieved)

    def _run_suite(self, prompts_path: str) -> list[TestExecutionResult]:
        tests = load_prompt_tests(prompts_path)
        results: list[TestExecutionResult] = []

        for test in tests:
            logger.info("Running test: %s", test.id)

            full_document = load_document(test.source_document)

            # Select context: RAG-retrieved chunks or full document.
            if self._use_rag_for(test):
                context = self._build_rag_context(full_document, test.question)
                logger.info(
                    "  [RAG] Using retrieved context (%d chars) for test %s",
                    len(context),
                    test.id,
                )
            else:
                context = full_document

            try:
                answer = self.provider.ask(test.question, context)
            except ProviderError as e:
                logger.error(
                    "  [ERROR] Provider failed for test %s: %s", test.id, e
                )
                results.append(
                    TestExecutionResult(
                        test_id=test.id,
                        question=test.question,
                        answer="",
                        evaluations=[],
                        error=str(e),
                    )
                )
                continue

            logger.debug("Question: %s", test.question)
            logger.debug("Answer:   %s", answer)

            evaluations = self.engine.evaluate(
                test_id=test.id,
                question=test.question,
                answer=answer,
                context=context,
            )

            for ev in evaluations:
                status = "PASSED" if ev.passed else "FAILED"
                logger.info(
                    "  [%s] %s  score=%.2f  reason=%s",
                    status, ev.metric, ev.score, ev.reason,
                )

            results.append(
                TestExecutionResult(
                    test_id=test.id,
                    question=test.question,
                    answer=answer,
                    evaluations=evaluations,
                )
            )

        return results
