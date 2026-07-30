from dataclasses import dataclass, field


@dataclass
class PromptTestCase:
    id: str
    question: str
    expected_answer: str
    source_document: str
    use_rag: bool = field(default=False)
    """
    When True this test case opts into RAG mode: the source document is
    chunked and only the top-k most relevant chunks are sent to the LLM
    as context instead of the full document text.  Requires ExecutionService
    to have been initialised with a RagConfig (rag_config.enabled need not
    be True globally — per-case use_rag always wins).
    """
