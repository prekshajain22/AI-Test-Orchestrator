from __future__ import annotations

import re
from dataclasses import dataclass, field

_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


@dataclass
class RetrievalMetrics:
    """
    Per-test retrieval quality report produced by ExecutionService when a
    test case runs in RAG mode.

    Keeps the retrieval layer measurable and auditable so the team can
    answer questions like "Which retriever reduces hallucinations?" or
    "Does BM25 improve faithfulness?" by comparing these metrics across
    runs.

    Attributes:
        retriever_name:    The retriever strategy used (tfidf, bm25, …).
        chunks_retrieved:  How many chunks the retriever returned (≤ top_k).
        chunk_scores:      Relevance scores, highest-first.
        average_similarity: Mean of chunk_scores.  0 → no overlap/similarity;
                            1 → perfect similarity.
        hit_at_k:          True if at least one retrieved chunk contains ≥ 50 %
                           of the expected-answer tokens.  Always True when
                           expected_answer is empty (unknown ground truth).
        mrr:               Mean Reciprocal Rank — 1/rank of the first
                           "relevant" chunk (same definition as hit_at_k).
                           0.0 if no relevant chunk found.
        context_coverage:  Fraction of expected-answer vocabulary tokens that
                           appear anywhere in the concatenated retrieved context.
                           1.0 when expected_answer is empty (no ground truth
                           to miss).
    """

    retriever_name: str
    chunks_retrieved: int
    chunk_scores: list[float] = field(default_factory=list)
    average_similarity: float = 0.0
    hit_at_k: bool = False
    mrr: float = 0.0
    context_coverage: float = 1.0

    @classmethod
    def compute(
        cls,
        retriever_name: str,
        retrieved,          # list[RetrievedChunk]
        expected_answer: str = "",
    ) -> "RetrievalMetrics":
        """
        Factory method — compute all metrics from a retrieval result list.

        Parameters
        ----------
        retriever_name:   Name tag for the retriever that produced the results.
        retrieved:        list[RetrievedChunk] returned by the retriever,
                          ordered highest-score first.
        expected_answer:  Ground-truth answer text used to judge relevance.
                          Pass "" when there is no ground truth.
        """
        if not retrieved:
            return cls(
                retriever_name=retriever_name,
                chunks_retrieved=0,
                chunk_scores=[],
                average_similarity=0.0,
                hit_at_k=False,
                mrr=0.0,
                context_coverage=1.0 if not expected_answer else 0.0,
            )

        scores = [rc.score for rc in retrieved]
        avg_sim = sum(scores) / len(scores)

        expected_tokens = _tokenize(expected_answer)

        # Build combined context token set for coverage.
        combined_context = " ".join(rc.chunk.text for rc in retrieved)
        context_tokens = _tokenize(combined_context)

        if not expected_tokens:
            # No ground truth → metrics that need it default to "unknown/ok".
            hit = True
            mrr_val = 1.0
            coverage = 1.0
        else:
            coverage = len(expected_tokens & context_tokens) / len(expected_tokens)

            # A chunk is "relevant" if it contains ≥ 50 % of the expected tokens.
            threshold = 0.5
            hit = False
            mrr_val = 0.0
            for rank, rc in enumerate(retrieved, start=1):
                chunk_tokens = _tokenize(rc.chunk.text)
                overlap = len(expected_tokens & chunk_tokens) / len(expected_tokens)
                if overlap >= threshold:
                    hit = True
                    if mrr_val == 0.0:      # first relevant chunk
                        mrr_val = 1.0 / rank
                    break

        return cls(
            retriever_name=retriever_name,
            chunks_retrieved=len(retrieved),
            chunk_scores=scores,
            average_similarity=round(avg_sim, 4),
            hit_at_k=hit,
            mrr=round(mrr_val, 4),
            context_coverage=round(coverage, 4),
        )
