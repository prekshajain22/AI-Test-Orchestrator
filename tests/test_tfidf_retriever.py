from ai_orchestrator.models import Chunk
from ai_orchestrator.retrievers.tfidf_retriever import TfidfRetriever


def _chunk(id_: str, heading: str, text: str) -> Chunk:
    return Chunk(id=id_, heading=heading, text=text)


def test_no_chunks_returns_empty_list():
    retriever = TfidfRetriever()
    assert retriever.retrieve("any question", [], top_k=3) == []


def test_returns_at_most_top_k_results():
    chunks = [
        _chunk("chunk_1", "A", "annual leave policy details here"),
        _chunk("chunk_2", "B", "sick leave policy details here"),
        _chunk("chunk_3", "C", "hybrid working policy details here"),
    ]
    retriever = TfidfRetriever()

    results = retriever.retrieve("What is the leave policy?", chunks, top_k=2)

    assert len(results) == 2


def test_more_relevant_chunk_scores_higher():
    chunks = [
        _chunk(
            "chunk_1",
            "Annual Leave",
            "Employees receive 25 days of annual leave per year.",
        ),
        _chunk(
            "chunk_2",
            "Security",
            "Confidential information must not be shared publicly.",
        ),
    ]
    retriever = TfidfRetriever()

    results = retriever.retrieve(
        "How many days of annual leave do employees get?", chunks, top_k=2
    )

    assert results[0].chunk.id == "chunk_1"
    assert results[0].score > results[1].score


def test_result_pairs_chunk_with_a_numeric_score():
    chunks = [_chunk("chunk_1", "A", "some sample text about leave policy")]
    retriever = TfidfRetriever()

    results = retriever.retrieve("leave policy", chunks, top_k=1)

    assert len(results) == 1
    assert results[0].chunk is chunks[0]
    assert isinstance(results[0].score, float)


def test_retrieval_is_deterministic():
    chunks = [
        _chunk("chunk_1", "A", "annual leave entitlement is 25 days"),
        _chunk("chunk_2", "B", "sick leave requires a doctor's note"),
    ]
    retriever = TfidfRetriever()

    result_1 = retriever.retrieve("What is the leave entitlement?", chunks, top_k=2)
    result_2 = retriever.retrieve("What is the leave entitlement?", chunks, top_k=2)

    assert [r.chunk.id for r in result_1] == [r.chunk.id for r in result_2]
    assert [r.score for r in result_1] == [r.score for r in result_2]


def test_question_with_no_matching_vocabulary_does_not_crash():
    chunks = [_chunk("chunk_1", "A", "annual leave entitlement details")]
    retriever = TfidfRetriever()

    results = retriever.retrieve("zzz qqq xyz", chunks, top_k=1)

    assert len(results) == 1
    assert results[0].score == 0.0


def test_end_to_end_with_real_sample_document_and_chunker():
    """Sanity check: chunk_document() output feeds directly into the retriever."""
    from pathlib import Path

    from ai_orchestrator.loaders.chunker import chunk_document

    text = Path("sample_data/documents/hybrid_working.md").read_text(encoding="utf-8")
    chunks = chunk_document(text)

    retriever = TfidfRetriever()
    results = retriever.retrieve(
        "What equipment does the company provide for home working?",
        chunks,
        top_k=3,
    )

    assert len(results) == 3
    headings = [r.chunk.heading for r in results]
    assert "Equipment and Workspace" in headings