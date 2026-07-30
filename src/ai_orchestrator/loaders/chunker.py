import re

from ai_orchestrator.models import Chunk

_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def chunk_document(text: str, max_chunk_chars: int = 800) -> list[Chunk]:
    """
    Split a document into Chunks for retrieval.

    This is the "R" (retrieval) half of RAG: instead of always handing an
    LLM the entire source document, a document is split into smaller,
    independently-retrievable passages up front. A later retrieval step
    (not implemented yet - see Chunk docstring) picks only the most
    relevant few of these to use as context for a given question.

    Splitting strategy, in order:
      1. If the document has markdown headings (#, ##, ...), split into
         one chunk per heading section. This keeps each chunk semantically
         coherent (e.g. one HR policy section per chunk) rather than
         cutting sentences in half at an arbitrary character count.
      2. Any section still longer than max_chunk_chars is further split
         by paragraph, so no single chunk becomes too large to
         meaningfully embed or retrieve later.
      3. Documents with no headings at all fall back to paragraph
         splitting directly.

    Args:
        text: The full document text.
        max_chunk_chars: Soft maximum size (in characters) for a single
            chunk before it gets split further by paragraph.

    Returns:
        A list of Chunks in document order. Empty/whitespace-only
        documents return an empty list.
    """
    sections = _split_by_headings(text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for heading, body in sections:
        for piece in _split_if_too_long(body, max_chunk_chars):
            piece = piece.strip()
            if not piece:
                continue
            chunk_index += 1
            chunks.append(
                Chunk(id=f"chunk_{chunk_index}", heading=heading, text=piece)
            )

    return chunks


def _split_by_headings(text: str) -> list[tuple[str | None, str]]:
    """Split text into (heading, section_text) pairs at markdown headings."""
    matches = list(_HEADING_PATTERN.finditer(text))

    if not matches:
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []

    # Any text before the first heading (e.g. a title with no # prefix)
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        section_text = f"{heading}\n{body}" if body else heading
        sections.append((heading, section_text))

    return sections


def _split_if_too_long(text: str, max_chars: int) -> list[str]:
    """Split an over-long section into smaller pieces along paragraph breaks."""
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    pieces: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces or [text]