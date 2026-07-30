from __future__ import annotations

import math

from ai_orchestrator.models import Chunk, RetrievedChunk
from ai_orchestrator.retrievers.base import BaseRetriever


class FaissRetriever(BaseRetriever):
    """
    Dense retriever using FAISS approximate-nearest-neighbour search.

    Builds a FAISS flat index (exact L2 search, no approximation) over
    sentence-transformer embeddings at query time, then converts L2 distance
    to a cosine-like score so results are comparable to other retrievers.

    For small corpora (< ~10 k chunks) a flat index is faster than training
    an IVF/HNSW index; swap to ``faiss.IndexHNSWFlat`` or
    ``faiss.IndexIVFFlat`` if you scale up.

    Dependencies: ``pip install faiss-cpu sentence-transformers``
    (use ``faiss-gpu`` on CUDA machines for large-scale indexing).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            import faiss  # type: ignore  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "FaissRetriever requires the 'faiss-cpu' package. "
                "Install it with:  pip install faiss-cpu sentence-transformers"
            ) from exc

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "FaissRetriever requires the 'sentence-transformers' package. "
                "Install it with:  pip install faiss-cpu sentence-transformers"
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

        import faiss  # type: ignore
        import numpy as np  # type: ignore

        texts = [chunk.text for chunk in chunks]
        all_texts = [question] + texts
        embeddings = self._model.encode(all_texts, convert_to_numpy=True).astype(
            "float32"
        )

        query_vec = embeddings[0:1]           # shape (1, dim)
        chunk_vecs = embeddings[1:]           # shape (n_chunks, dim)

        dim = chunk_vecs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(chunk_vecs)

        k = min(top_k, len(chunks))
        distances, indices = index.search(query_vec, k)

        # Convert L2 distance → similarity score in [0, 1].
        # distance = 2(1 − cos_sim) for unit-normed vectors, so:
        # cos_sim = 1 − distance/2 (clamped to [0, 1]).
        results: list[RetrievedChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = max(0.0, 1.0 - float(dist) / 2.0)
            results.append(
                RetrievedChunk(chunk=chunks[int(idx)], score=similarity)
            )

        # Already ordered by distance (ascending), so reverse for score descending.
        results.sort(key=lambda rc: rc.score, reverse=True)
        return results
