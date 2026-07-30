from __future__ import annotations

import logging
from typing import Optional

from ai_orchestrator.config.loader import RagConfig
from ai_orchestrator.evaluators.engine import EvaluationEngine
from ai_orchestrator.evaluators import EvaluationFactory
from ai_orchestrator.loaders import load_document, load_prompt_tests
from ai_orchestrator.loaders.chunker import chunk_document
from ai_orchestrator.models import RetrievalMetrics, TestExecutionResult
from ai_orchestrator.providers import ProviderFactory
from ai_orchestrator.providers.base import ProviderError
from ai_orchestrator.retrievers import TfidfRetriever, BM25Retriever
from ai_orchestrator.retrievers.base import BaseRetriever

logger = logging.getLogger(__name__)


def _build_retriever(retriever_name: str) -> BaseRetriever:
    """
    Instantiate a retriever by name.

    Pure-Python retrievers (tfidf, bm25) are imported eagerly.
    Heavy-dependency retrievers (sentence_transformer, faiss, chroma) are
    imported lazily here so that the application starts successfully even
    when the optional packages are not installed — the ImportError is only
    raised if you actually try to USE one of those retrievers.
    """
    if retriever_name == "tfidf":
        return TfidfRetriever()

    if retriever_name == "bm25":
        return BM25Retriever()

    if retriever_name == "sentence_transformer":
        from ai_orchestrator.retrievers.sentence_transformer_retriever import (
            SentenceTransformerRetriever,
        )
        return SentenceTransformerRetriever()

    if retriever_name == "faiss":
        from ai_orchestrator.retrievers.faiss_retriever import FaissRetriever
        return FaissRetriever()

    if retriever_name == "chroma":
        from ai_orchestrator.retrievers.chroma_retriever import ChromaRetriever
        return ChromaRetriever()

    available = ["tfidf", "bm25", "sentence_transformer", "faiss", "chroma"]
    raise ValueError(
        f"Unknown retriever: '{retriever_name}'. Available: {available}"
    )


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

    Supported retrievers (``rag_config.retriever``):
      - ``tfidf``               — TF-IDF cosine similarity (pure Python, default)
      - ``bm25``                — Okapi BM25 (pure Python)
      - ``sentence_transformer``— Dense embeddings via sentence-transformers
                                  (``pip install sentence-transformers``)
      - ``faiss``               — FAISS ANN + sentence-transformers
                                  (``pip install faiss-cpu sentence-transformers``)
      - ``chroma``              — ChromaDB in-memory vector store
                                  (``pip install chromadb``)

    Configuration loading is the caller's responsibility (see
    ai_orchestrator.config.loader.load_execution_config and
    ai_orchestrator.runners.runner.TestRunner).
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
        self._retriever: BaseRetriever = _build_retriever(
            self._rag_config.retriever
        )

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

    def _build_rag_context(
        self, document_text: str, question: str, expected_answer: str
    ) -> tuple[str, RetrievalMetrics]:
        """
        Chunk *document_text*, retrieve top-k most relevant chunks, join
        their text, and return ``(context_str, RetrievalMetrics)``.

        Falls back to the full document if chunking or retrieval returns
        nothing, so the LLM call is never left without context.
        """
        retriever_name = self._rag_config.retriever
        chunks = chunk_document(document_text)

        if not chunks:
            logger.warning("chunk_document returned no chunks — using full document.")
            metrics = RetrievalMetrics.compute(
                retriever_name=retriever_name,
                retrieved=[],
                expected_answer=expected_answer,
            )
            return document_text, metrics

        top_k = self._rag_config.top_k
        retrieved = self._retriever.retrieve(question, chunks, top_k=top_k)

        if not retrieved:
            logger.warning(
                "Retriever returned no results for question %r — "
                "falling back to full document.",
                question,
            )
            metrics = RetrievalMetrics.compute(
                retriever_name=retriever_name,
                retrieved=[],
                expected_answer=expected_answer,
            )
            return document_text, metrics

        metrics = RetrievalMetrics.compute(
            retriever_name=retriever_name,
            retrieved=retrieved,
            expected_answer=expected_answer,
        )
        context = "\n\n".join(rc.chunk.text for rc in retrieved)
        return context, metrics

    def _log_retrieval_metrics(self, test_id: str, metrics: RetrievalMetrics) -> None:
        """Emit a structured INFO block matching the user-facing report format."""
        scores_str = "  ".join(f"{s:.2f}" for s in metrics.chunk_scores)
        relevant = "YES" if metrics.hit_at_k else "NO"
        logger.info(
            "  [RETRIEVAL] test=%s  retriever=%s  chunks=%d  "
            "scores=[%s]  relevant=%s  avg_sim=%.2f  coverage=%.0f%%  mrr=%.2f",
            test_id,
            metrics.retriever_name,
            metrics.chunks_retrieved,
            scores_str,
            relevant,
            metrics.average_similarity,
            metrics.context_coverage * 100,
            metrics.mrr,
        )

    def _run_suite(self, prompts_path: str) -> list[TestExecutionResult]:
        tests = load_prompt_tests(prompts_path)
        results: list[TestExecutionResult] = []

        for test in tests:
            logger.info("Running test: %s", test.id)

            full_document = load_document(test.source_document)
            retrieval_metrics: Optional[RetrievalMetrics] = None

            # Select context: RAG-retrieved chunks or full document.
            if self._use_rag_for(test):
                context, retrieval_metrics = self._build_rag_context(
                    full_document, test.question, test.expected_answer
                )
                self._log_retrieval_metrics(test.id, retrieval_metrics)
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
                        retrieval_metrics=retrieval_metrics,
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
                    retrieval_metrics=retrieval_metrics,
                )
            )

        return results
