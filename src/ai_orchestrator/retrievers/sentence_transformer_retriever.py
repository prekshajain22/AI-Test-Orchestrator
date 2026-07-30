from __future__ import annotations

import math

from ai_orchestrator.models import Chunk, RetrievedChunk
from ai_orchestrator.retrievers.base import BaseRetriever


class SentenceTransformerRetriever(BaseRetriever):
    """
    Dense retriever using sentence-transformers embeddings + cosine similarity.

    Unlike TF-IDF or BM25 (which rely on exact keyword overlap), this
    retriever converts both the question and each chunk into dense semantic
    vectors.  Chunks are ranked by how semantically similar they are to the
    question, so paraphrases and synonyms score highly even when they share
    no words.

    Dependency: ``pip install sentence-transformers``
    Model: defaults to ``all-MiniLM-L6-v2`` — small (80 MB), fast, and
    competitive on retrieval benchmarks.  Override via ``model_name``.

    Embeddings are computed lazily on first call and cached for the lifetime
    of the retriever instance (stateless between different chunk sets: a new
    encoding is done each call because chunks vary per document).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "SentenceTransformerRetriever requires the 'sentence-transformers' "
                "package.  Install it with:  pip install sentence-transformers"
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def retrieve(
        self,
        question: str,
        chunks: list[Chunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        # Encode question and all chunk texts in one batched call for speed.
        all_texts = [question] + texts
        embeddings = self._model.encode(all_texts, convert_to_numpy=True)

        query_vec = embeddings[0]
        chunk_vecs = embeddings[1:]

        scored = [
            RetrievedChunk(
                chunk=chunk,
                score=float(self._cosine(query_vec, chunk_vec)),
            )
            for chunk, chunk_vec in zip(chunks, chunk_vecs)
        ]

        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _cosine(a, b) -> float:
        """Cosine similarity between two numpy vectors."""
        dot = float((a * b).sum())
        norm_a = float(math.sqrt((a * a).sum()))
        norm_b = float(math.sqrt((b * b).sum()))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (norm_a * norm_b)
