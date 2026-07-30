from abc import ABC, abstractmethod

from ai_orchestrator.models import Chunk, RetrievedChunk


class BaseRetriever(ABC):
    """
    Base interface for all retrieval strategies.

    Given a question and the full list of chunks a document was split
    into (see loaders.chunker.chunk_document), a retriever selects and
    ranks the top_k chunks most relevant to that question. This is the
    "R" (retrieval) half of RAG: only the retrieved chunks - not the
    whole document - should end up as context sent to an LLM provider.

    Same Strategy pattern as LLMClient (providers/base.py) and
    BaseEvaluator (evaluators/base.py): swap the implementation without
    changing anything that calls it.
    """

    @abstractmethod
    def retrieve(
        self,
        question: str,
        chunks: list[Chunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Return the top_k chunks most relevant to the question, ordered
        highest-score first.
        """
        pass