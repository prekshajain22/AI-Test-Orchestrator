import math
import re
from collections import Counter

from ai_orchestrator.models import Chunk, RetrievedChunk
from ai_orchestrator.retrievers.base import BaseRetriever

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")

# Okapi BM25 hyperparameters — industry-standard defaults.
_K1 = 1.5   # term-frequency saturation: higher → more weight on raw TF
_B = 0.75   # document-length normalisation: 1 = full, 0 = none


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever(BaseRetriever):
    """
    Retrieves the most relevant chunks using Okapi BM25.

    BM25 addresses two known TF-IDF weaknesses:
    1. Term-frequency saturation — repeating a word 100× doesn't score 100×
       better than repeating it 10×.
    2. Document-length normalisation — longer chunks don't dominate just
       because they contain more word occurrences.

    Implemented with pure Python (no external dependencies) for the same
    reasons as TfidfRetriever: deterministic, lightweight, no ML setup.

    Formula:  score(D, Q) =
        Σ_{q ∈ Q} IDF(q) · (tf(q, D) · (k1 + 1))
                           / (tf(q, D) + k1 · (1 − b + b · |D| / avgdl))

    IDF(q) = log((N − df(q) + 0.5) / (df(q) + 0.5) + 1)   [Robertson IDF]
    """

    def retrieve(
        self,
        question: str,
        chunks: list[Chunk],
        top_k: int = 3,
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        documents = [_tokenize(chunk.text) for chunk in chunks]
        query_tokens = _tokenize(question)

        if not query_tokens:
            # No scoreable query — return all chunks with score 0, top_k limited.
            return [RetrievedChunk(chunk=c, score=0.0) for c in chunks][:top_k]

        avgdl = sum(len(d) for d in documents) / len(documents)
        idf = self._compute_idf(documents, len(documents))

        scored = [
            RetrievedChunk(
                chunk=chunk,
                score=self._bm25_score(query_tokens, doc, idf, avgdl),
            )
            for chunk, doc in zip(chunks, documents)
        ]

        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _compute_idf(documents: list[list[str]], n_docs: int) -> dict[str, float]:
        """Robertson IDF with +1 smoothing to prevent negative values."""
        df: Counter = Counter()
        for doc in documents:
            for term in set(doc):
                df[term] += 1

        return {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in df.items()
        }

    @staticmethod
    def _bm25_score(
        query_tokens: list[str],
        doc_tokens: list[str],
        idf: dict[str, float],
        avgdl: float,
    ) -> float:
        tf = Counter(doc_tokens)
        doc_len = len(doc_tokens)
        score = 0.0

        for term in set(query_tokens):
            if term not in idf:
                continue
            tf_val = tf.get(term, 0)
            numerator = tf_val * (_K1 + 1)
            denominator = tf_val + _K1 * (1 - _B + _B * doc_len / avgdl)
            score += idf[term] * numerator / denominator

        return score
