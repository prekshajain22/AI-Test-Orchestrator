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

LLM_JUDGE = """You are an expert AI evaluator. Score the following AI answer on four dimensions.

Context (source document excerpt):
{context}

Question:
{question}

AI Answer:
{answer}

Score each dimension from 0.0 to 1.0 (two decimal places).

Correctness:   Does the answer contain factually correct information relative to the context?
Completeness:  Does the answer address all key aspects of the question?
Groundedness:  Is the answer fully supported by the context (no information invented beyond it)?
Helpfulness:   Would a user find this answer useful and actionable?

Respond ONLY with a JSON object in this exact format (no markdown, no extra text):
{{"correctness": 0.00, "completeness": 0.00, "groundedness": 0.00, "helpfulness": 0.00}}"""


def render_llm_judge_prompt(question: str, context: str, answer: str) -> str:
    """Render the LLM-as-a-Judge scoring prompt."""
    return LLM_JUDGE.format(context=context, question=question, answer=answer)


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
