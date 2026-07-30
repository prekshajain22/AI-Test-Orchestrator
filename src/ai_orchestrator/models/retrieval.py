from dataclasses import dataclass

from ai_orchestrator.models.chunk import Chunk


@dataclass
class RetrievedChunk:
    """
    One chunk a retriever selected for a given question, paired with the
    relevance score it assigned.

    Keeping the score alongside the chunk (rather than returning bare
    Chunks) is what makes retrieval measurable later - e.g. a future
    Ragas-style "context precision" metric needs to know not just which
    chunks were retrieved, but how confident the retriever was in each.
    """

    chunk: Chunk
    score: float