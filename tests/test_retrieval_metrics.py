import pytest

from ai_orchestrator.models import Chunk, RetrievedChunk, RetrievalMetrics


def _rc(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=Chunk(id=chunk_id, heading=None, text=text),
        score=score,
    )


# ── Basic construction via compute() ─────────────────────────────────────────

def test_empty_retrieved_list_returns_zero_metrics():
    m = RetrievalMetrics.compute(
        retriever_name="tfidf",
        retrieved=[],
        expected_answer="Employees get 25 days annual leave.",
    )
    assert m.chunks_retrieved == 0
    assert m.chunk_scores == []
    assert m.average_similarity == 0.0
    assert m.hit_at_k is False
    assert m.mrr == 0.0
    assert m.context_coverage == 0.0


def test_empty_retrieved_no_expected_answer_coverage_defaults_zero():
    m = RetrievalMetrics.compute(
        retriever_name="bm25",
        retrieved=[],
        expected_answer="",
    )
    # No expected answer → no ground truth to miss.
    assert m.context_coverage == 1.0


def test_chunks_retrieved_count():
    retrieved = [
        _rc("c1", "annual leave is 25 days", 0.9),
        _rc("c2", "sick leave requires certificate", 0.7),
    ]
    m = RetrievalMetrics.compute("tfidf", retrieved)
    assert m.chunks_retrieved == 2


def test_chunk_scores_are_recorded_in_order():
    retrieved = [
        _rc("c1", "text one", 0.91),
        _rc("c2", "text two", 0.84),
        _rc("c3", "text three", 0.77),
    ]
    m = RetrievalMetrics.compute("tfidf", retrieved)
    assert m.chunk_scores == pytest.approx([0.91, 0.84, 0.77])


def test_average_similarity_is_mean_of_scores():
    retrieved = [
        _rc("c1", "text one", 0.9),
        _rc("c2", "text two", 0.7),
    ]
    m = RetrievalMetrics.compute("tfidf", retrieved)
    assert m.average_similarity == pytest.approx(0.8, abs=1e-4)


# ── hit@k ─────────────────────────────────────────────────────────────────────

def test_hit_at_k_true_when_chunk_contains_expected_answer_tokens():
    # expected has 5 tokens; chunk_1 contains 4 of them (80 % ≥ 50 %)
    retrieved = [
        _rc(
            "c1",
            "Employees receive 25 days of annual leave per year.",
            0.9,
        ),
    ]
    m = RetrievalMetrics.compute(
        "tfidf",
        retrieved,
        expected_answer="25 days annual leave",
    )
    assert m.hit_at_k is True


def test_hit_at_k_false_when_no_chunk_contains_enough_expected_tokens():
    retrieved = [
        _rc("c1", "Remote working requires a stable internet connection.", 0.5),
    ]
    m = RetrievalMetrics.compute(
        "tfidf",
        retrieved,
        expected_answer="annual leave 25 days paid holiday entitlement",
    )
    assert m.hit_at_k is False


def test_hit_at_k_true_when_no_expected_answer_provided():
    """No ground truth → hit defaults to True (unknown, not failed)."""
    retrieved = [_rc("c1", "some text here", 0.5)]
    m = RetrievalMetrics.compute("tfidf", retrieved, expected_answer="")
    assert m.hit_at_k is True


# ── MRR ──────────────────────────────────────────────────────────────────────

def test_mrr_is_1_when_first_chunk_is_relevant():
    retrieved = [
        _rc("c1", "Employees receive 25 days annual leave.", 0.9),
        _rc("c2", "Confidential data must be protected.", 0.6),
    ]
    m = RetrievalMetrics.compute(
        "tfidf", retrieved, expected_answer="25 days annual leave"
    )
    assert m.mrr == pytest.approx(1.0)


def test_mrr_is_0_when_no_chunk_is_relevant():
    retrieved = [
        _rc("c1", "Remote desk setup policy.", 0.4),
        _rc("c2", "Internet security guidelines.", 0.3),
    ]
    m = RetrievalMetrics.compute(
        "tfidf",
        retrieved,
        expected_answer="annual leave 25 days paid holiday entitlement",
    )
    assert m.mrr == 0.0


# ── context_coverage ──────────────────────────────────────────────────────────

def test_context_coverage_full_when_all_tokens_present():
    retrieved = [
        _rc(
            "c1",
            "Employees are entitled to 25 days of annual leave per year.",
            0.9,
        ),
    ]
    m = RetrievalMetrics.compute(
        "tfidf",
        retrieved,
        expected_answer="25 days annual leave",
    )
    assert m.context_coverage == pytest.approx(1.0)


def test_context_coverage_partial_when_some_tokens_missing():
    retrieved = [
        _rc("c1", "Employees must notify their manager.", 0.7),
    ]
    m = RetrievalMetrics.compute(
        "tfidf",
        retrieved,
        expected_answer="notify manager about sick leave absence",
    )
    # "notify" and "manager" appear; "about", "sick", "leave", "absence" may not.
    assert 0.0 < m.context_coverage < 1.0


def test_context_coverage_across_multiple_chunks():
    """Tokens spread across two chunks should still contribute to coverage."""
    retrieved = [
        _rc("c1", "Employees are entitled to 25 days.", 0.9),
        _rc("c2", "Annual leave must be agreed with the manager.", 0.7),
    ]
    m = RetrievalMetrics.compute(
        "tfidf",
        retrieved,
        expected_answer="25 days annual leave",
    )
    assert m.context_coverage == pytest.approx(1.0)


# ── retriever_name propagation ────────────────────────────────────────────────

def test_retriever_name_is_stored():
    retrieved = [_rc("c1", "text", 0.5)]
    m = RetrievalMetrics.compute("bm25", retrieved, expected_answer="")
    assert m.retriever_name == "bm25"


# ── Integration: metrics stored on TestExecutionResult ───────────────────────

def test_execution_result_stores_retrieval_metrics():
    from ai_orchestrator.models import TestExecutionResult

    rm = RetrievalMetrics.compute("tfidf", [_rc("c1", "text", 0.8)])
    result = TestExecutionResult(
        test_id="t1",
        question="Q?",
        answer="A",
        evaluations=[],
        retrieval_metrics=rm,
    )
    assert result.retrieval_metrics is rm
    assert result.retrieval_metrics.retriever_name == "tfidf"


def test_execution_result_retrieval_metrics_defaults_to_none():
    from ai_orchestrator.models import TestExecutionResult

    result = TestExecutionResult(
        test_id="t1",
        question="Q?",
        answer="A",
        evaluations=[],
    )
    assert result.retrieval_metrics is None
