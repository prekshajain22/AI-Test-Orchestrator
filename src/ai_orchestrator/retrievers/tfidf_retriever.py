import math
import re
from collections import Counter

from ai_orchestrator.models import Chunk, RetrievedChunk
from ai_orchestrator.retrievers.base import BaseRetriever

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class TfidfRetriever(BaseRetriever):
    """
    Retrieves the most relevant chunks for a question using classic
    TF-IDF term weighting + cosine similarity.

    Implemented with pure Python (no scikit-learn / ML dependency) so it
    stays lightweight and fully deterministic: the same question and
    chunks always produce the same ranking and scores.
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

        idf = self._compute_idf(documents)
        doc_vectors = [self._tfidf_vector(doc, idf) for doc in documents]
        query_vector = self._tfidf_vector(query_tokens, idf)

        scored = [
            RetrievedChunk(
                chunk=chunk,
                score=self._cosine_similarity(query_vector, doc_vector),
            )
            for chunk, doc_vector in zip(chunks, doc_vectors)
        ]

        scored.sort(key=lambda rc: rc.score, reverse=True)
        return scored[:top_k]

    @staticmethod
    def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
        """Smoothed inverse document frequency, one entry per vocabulary term."""
        n_docs = len(documents)
        document_frequency: Counter = Counter()

        for doc in documents:
            for term in set(doc):
                document_frequency[term] += 1

        return {
            term: math.log((1 + n_docs) / (1 + freq)) + 1
            for term, freq in document_frequency.items()
        }

    @staticmethod
    def _tfidf_vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
        """TF-IDF weighted term vector for a single tokenized document/query."""
        if not tokens:
            return {}

        term_counts = Counter(tokens)
        total_terms = len(tokens)

        return {
            term: (count / total_terms) * idf.get(term, 0.0)
            for term, count in term_counts.items()
        }

    @staticmethod
    def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0

        shared_terms = set(vec_a) & set(vec_b)
        dot_product = sum(vec_a[term] * vec_b[term] for term in shared_terms)

        norm_a = math.sqrt(sum(weight * weight for weight in vec_a.values()))
        norm_b = math.sqrt(sum(weight * weight for weight in vec_b.values()))

        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0

        return dot_product / (norm_a * norm_b)
