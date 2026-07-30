from ai_orchestrator.models import Chunk
from ai_orchestrator.retrievers.bm25_retriever import BM25Retriever


def _chunk(id_: str, heading: str, text: str) -> Chunk:
    return Chunk(id=id_, heading=heading, text=text)


# ── Basic contract tests (mirrors test_tfidf_retriever.py) ────────────────────

def test_no_chunks_returns_empty_list():
    retriever = BM25Retriever()
    assert retriever.retrieve("any question", [], top_k=3) == []


def test_returns_at_most_top_k_results():
    chunks = [
        _chunk("c1", "A", "annual leave policy details here"),
        _chunk("c2", "B", "sick leave policy details here"),
        _chunk("c3", "C", "hybrid working policy details here"),
    ]
    results = BM25Retriever().retrieve("What is the leave policy?", chunks, top_k=2)
    assert len(results) == 2


def test_more_relevant_chunk_scores_higher():
    chunks = [
        _chunk("c1", "Annual", "Employees receive 25 days of annual leave per year."),
        _chunk("c2", "Security", "Confidential information must not be shared."),
    ]
    results = BM25Retriever().retrieve(
        "How many days of annual leave do employees get?", chunks, top_k=2
    )
    assert results[0].chunk.id == "c1"
    assert results[0].score > results[1].score


def test_result_pairs_chunk_with_a_numeric_score():
    chunks = [_chunk("c1", "A", "some sample text about leave policy")]
    results = BM25Retriever().retrieve("leave policy", chunks, top_k=1)
    assert len(results) == 1
    assert results[0].chunk is chunks[0]
    assert isinstance(results[0].score, float)


def test_retrieval_is_deterministic():
    chunks = [
        _chunk("c1", "A", "annual leave entitlement is 25 days"),
        _chunk("c2", "B", "sick leave requires a doctor's note"),
    ]
    retriever = BM25Retriever()
    r1 = retriever.retrieve("What is the leave entitlement?", chunks, top_k=2)
    r2 = retriever.retrieve("What is the leave entitlement?", chunks, top_k=2)
    assert [r.chunk.id for r in r1] == [r.chunk.id for r in r2]
    assert [r.score for r in r1] == [r.score for r in r2]


def test_question_with_no_matching_vocabulary_returns_zero_scores():
    chunks = [_chunk("c1", "A", "annual leave entitlement details")]
    results = BM25Retriever().retrieve("zzz qqq xyz", chunks, top_k=1)
    # No shared vocab → BM25 scores all 0.
    assert len(results) == 1
    assert results[0].score == 0.0


# ── BM25-specific: TF saturation ─────────────────────────────────────────────

def test_bm25_tf_saturation():
    """
    BM25 must NOT score a chunk that repeats a keyword 20× proportionally
    higher than a chunk that mentions it 2×.  TF-IDF would; BM25 should not.
    This verifies the k1 saturation parameter is working.
    """
    # chunk_1 repeats "leave" 20 times — raw TF is very high.
    chunk_1 = _chunk("c1", "A", "leave " * 20)
    # chunk_2 mentions "leave" only twice but also matches "annual" and "days".
    chunk_2 = _chunk("c2", "B", "annual leave entitlement is 25 days of paid leave")

    results = BM25Retriever().retrieve(
        "How many annual leave days?", [chunk_1, chunk_2], top_k=2
    )

    # chunk_2 should score higher because it matches more distinct query terms.
    assert results[0].chunk.id == "c2"


def test_bm25_document_length_normalisation():
    """
    A very long irrelevant document should not outrank a short relevant one
    just because it happens to contain the query term once.
    """
    short_relevant = _chunk(
        "c1", "A", "employees are entitled to 25 days annual leave"
    )
    # Very long document that mentions "leave" once in a sea of unrelated words.
    long_irrelevant = _chunk(
        "c2",
        "B",
        ("unrelated words about nothing important " * 30) + " leave ",
    )

    results = BM25Retriever().retrieve(
        "How many days annual leave?", [short_relevant, long_irrelevant], top_k=2
    )
    assert results[0].chunk.id == "c1"


def test_end_to_end_with_real_sample_document_and_chunker():
    """Sanity check: chunk_document() output feeds into BM25Retriever."""
    from pathlib import Path
    from ai_orchestrator.loaders.chunker import chunk_document

    text = Path("sample_data/documents/sick_leave.md").read_text(encoding="utf-8")
    chunks = chunk_document(text)

    results = BM25Retriever().retrieve(
        "When should an employee notify their manager about sick leave?",
        chunks,
        top_k=3,
    )

    assert len(results) >= 1
    # The top result should come from a section about notification.
    top_text = results[0].chunk.text.lower()
    assert "notify" in top_text or "notification" in top_text or "manager" in top_text
