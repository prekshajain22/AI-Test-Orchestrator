from ai_orchestrator.prompts import render_qa_prompt
from ai_orchestrator.prompts.templates import QA_WITH_CONTEXT


def test_render_qa_prompt_includes_question_and_context():
    prompt = render_qa_prompt(
        question="What is the leave policy?",
        context="Employees get 25 days annual leave.",
    )
    assert "What is the leave policy?" in prompt
    assert "Employees get 25 days annual leave." in prompt


def test_render_qa_prompt_matches_template_exactly():
    """
    Prompts should be testable independent of any LLM call: given a fixed
    template, question and context, the rendered output must be exact.
    """
    question = "Q?"
    context = "C."
    expected = QA_WITH_CONTEXT.format(context=context, question=question)

    assert render_qa_prompt(question, context) == expected


def test_render_qa_prompt_is_deterministic():
    result_1 = render_qa_prompt("Q", "C")
    result_2 = render_qa_prompt("Q", "C")
    assert result_1 == result_2
