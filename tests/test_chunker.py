from ai_orchestrator.loaders.chunker import chunk_document


def test_empty_document_returns_no_chunks():
    assert chunk_document("") == []
    assert chunk_document("   \n\n  ") == []


def test_document_with_no_headings_becomes_a_single_chunk():
    text = "Just a plain paragraph with no markdown headings at all."
    chunks = chunk_document(text)

    assert len(chunks) == 1
    assert chunks[0].heading is None
    assert chunks[0].text == text


def test_document_is_split_one_chunk_per_heading():
    text = (
        "# Hybrid Working Policy\n\n"
        "## Eligibility\n"
        "Employees must have manager approval.\n\n"
        "## Work Schedule\n"
        "Employees should coordinate a regular hybrid schedule.\n"
    )
    chunks = chunk_document(text)

    headings = [c.heading for c in chunks]
    assert "Eligibility" in headings
    assert "Work Schedule" in headings
    # top-level title heading also becomes its own chunk
    assert "Hybrid Working Policy" in headings


def test_each_chunk_contains_only_its_own_section_content():
    text = (
        "## Eligibility\n"
        "Eligibility text here.\n\n"
        "## Work Schedule\n"
        "Schedule text here.\n"
    )
    chunks = chunk_document(text)

    eligibility_chunk = next(c for c in chunks if c.heading == "Eligibility")
    schedule_chunk = next(c for c in chunks if c.heading == "Work Schedule")

    assert "Eligibility text here" in eligibility_chunk.text
    assert "Schedule text here" not in eligibility_chunk.text
    assert "Schedule text here" in schedule_chunk.text
    assert "Eligibility text here" not in schedule_chunk.text


def test_chunk_ids_are_unique_and_sequential():
    text = "## A\ntext a\n\n## B\ntext b\n\n## C\ntext c\n"
    chunks = chunk_document(text)

    ids = [c.id for c in chunks]
    assert ids == [f"chunk_{i}" for i in range(1, len(chunks) + 1)]
    assert len(set(ids)) == len(ids)  # all unique


def test_long_section_is_split_further_by_paragraph():
    long_paragraph_1 = "Sentence one. " * 30   # ~450 chars
    long_paragraph_2 = "Sentence two. " * 30   # ~450 chars
    text = f"## Long Section\n{long_paragraph_1}\n\n{long_paragraph_2}\n"

    chunks = chunk_document(text, max_chunk_chars=500)

    long_section_chunks = [c for c in chunks if c.heading == "Long Section"]
    assert len(long_section_chunks) > 1
    for chunk in long_section_chunks:
        assert len(chunk.text) <= 550  # allow small overhead from heading text


def test_short_section_is_not_split():
    text = "## Short Section\nJust one short sentence.\n"
    chunks = chunk_document(text, max_chunk_chars=800)

    assert len(chunks) == 1


def test_real_sample_document_chunks_cleanly():
    """Sanity check against one of the project's real sample documents."""
    from pathlib import Path

    doc_path = Path("sample_data/documents/hybrid_working.md")
    text = doc_path.read_text(encoding="utf-8")

    chunks = chunk_document(text)

    assert len(chunks) > 1
    # every chunk should be non-empty and traceable to a heading
    for chunk in chunks:
        assert chunk.text.strip()
        assert chunk.id.startswith("chunk_")