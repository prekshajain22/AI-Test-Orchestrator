from __future__ import annotations

from ai_orchestrator.models import Chunk, RetrievedChunk
from ai_orchestrator.retrievers.base import BaseRetriever


class ChromaRetriever(BaseRetriever):
    """
    Dense retriever backed by ChromaDB's in-process vector store.

    ChromaDB handles embedding generation, vector storage, and ANN search
    in a single call.  This implementation uses an ephemeral in-memory
    client (no persistence, no server required), so every call to
    ``retrieve`` re-indexes the supplied chunks.  For a persistent store,
    swap ``chromadb.Client()`` for ``chromadb.PersistentClient(path=...)``.

    Embedding function: ``all-MiniLM-L6-v2`` via ChromaDB's built-in
    ``SentenceTransformerEmbeddingFunction`` (same model as
    SentenceTransformerRetriever, so scores are directly comparable).

    Score returned: ChromaDB returns L2 distance; we convert to a
    similarity in [0, 1] via ``1 − distance/2`` so the scale matches
    the other retrievers.

    Dependency: ``pip install chromadb``
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            import chromadb  # type: ignore  # noqa: F401
            from chromadb.utils.embedding_functions import (  # type: ignore
                SentenceTransformerEmbeddingFunction,
            )
        except ImportError as exc:
            raise ImportError(
                "ChromaRetriever requires the 'chromadb' package. "
                "Install it with:  pip install chromadb"
            ) from exc

        self.model_name = model_name
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )

    def retrieve(
        self,
        question: str,
        chunks: list[Chunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        import chromadb  # type: ignore

        # Fresh ephemeral client per call — avoids state leaking between tests
        # and different document sets.
        client = chromadb.Client()
        collection = client.create_collection(
            name="rag_chunks",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "l2"},
        )

        collection.add(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
        )

        k = min(top_k, len(chunks))
        result = collection.query(
            query_texts=[question],
            n_results=k,
            include=["distances"],
        )

        distances = result["distances"][0]    # list[float], ascending L2 dist
        retrieved_ids = result["ids"][0]      # list[str]

        # Build a lookup so we can match ids → original Chunk objects.
        chunk_by_id = {chunk.id: chunk for chunk in chunks}

        scored = [
            RetrievedChunk(
                chunk=chunk_by_id[cid],
                score=max(0.0, 1.0 - float(dist) / 2.0),
            )
            for cid, dist in zip(retrieved_ids, distances)
        ]

        # Sort highest similarity first (Chroma returns lowest-distance first,
        # but after the score inversion we sort descending to be safe).
        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored
