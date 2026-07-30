"""
Shared prompt templates for LLM providers.

Both providers (GeminiProvider, HuggingFaceClient) build a prompt for the
same "answer a question using only the provided context" task. Previously
each provider hardcoded its own f-string, which meant:
  - the two providers could silently drift apart in wording
  - there was no single place to test/version the prompt
  - swapping in a new prompt (e.g. for A/B testing hallucination rates)
    required touching provider code

This module is the single source of truth for that prompt. Providers call
`render_qa_prompt()` instead of building their own f-string.
"""

QA_WITH_CONTEXT = """Answer the question using only the provided context.
If the answer is not present in the context, say:
"I cannot find that information in the provided policy."

Context:
{context}

Question:
{question}

Answer:"""


def render_qa_prompt(question: str, context: str) -> str:
    """
    Render the standard question-answering prompt.

    Args:
        question: The user's question.
        context: The source document content to answer from.

    Returns:
        The fully rendered prompt string to send to an LLM provider.
    """
    return QA_WITH_CONTEXT.format(context=context, question=question)
