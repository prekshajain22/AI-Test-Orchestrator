from dataclasses import dataclass


@dataclass
class Chunk:
    """
    A single retrievable passage extracted from a source document.

    RAG ("Retrieval-Augmented Generation") works by splitting a document
    into smaller passages like this one, so that only the few most
    relevant chunks are retrieved and handed to the LLM as context -
    instead of always passing the entire document, which is what
    load_document() does today.

    Attributes:
        id: A stable identifier for this chunk within its document
            (e.g. "chunk_1"). Useful for tracing which chunk was
            retrieved for a given answer.
        heading: The markdown heading this chunk falls under, if the
            source document had one (e.g. "Eligibility"). None if the
            chunk came from a document with no headings.
        text: The chunk's actual passage text.
    """

    id: str
    heading: str | None
    text: str