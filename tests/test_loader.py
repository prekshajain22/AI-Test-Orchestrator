import textwrap

import pytest

from ai_orchestrator.loaders.prompt_loader import load_prompt_tests
from ai_orchestrator.loaders.document_loader import load_document
from ai_orchestrator.config.loader import (
    load_test_suite,
    load_evaluation_config,
    TestSuiteConfig,
    EvaluationConfig,
)


PROMPT_YAML = textwrap.dedent("""\
    tests:
      - id: t1
        question: "What is the leave policy?"
        source_document: "docs/leave.md"
        expected_answer: "25 days"
      - id: t2
        question: "Hybrid working?"
        source_document: "docs/hybrid.md"
""")

TEST_SUITE_YAML = textwrap.dedent("""\
    provider: gemini
    tests:
      - sample_data/prompts/hr_questions.yaml
""")

EVAL_YAML = textwrap.dedent("""\
    evaluators:
      - hallucination
      - relevance
""")


# ── Prompt loader ──────────────────────────────────────────────

def test_load_prompt_tests_returns_correct_count(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text(PROMPT_YAML)
    tests = load_prompt_tests(str(p))
    assert len(tests) == 2


def test_load_prompt_tests_fields(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text(PROMPT_YAML)
    tests = load_prompt_tests(str(p))
    assert tests[0].id == "t1"
    assert tests[0].question == "What is the leave policy?"
    assert tests[0].source_document == "docs/leave.md"
    assert tests[0].expected_answer == "25 days"


def test_load_prompt_tests_missing_expected_answer_defaults_empty(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text(PROMPT_YAML)
    tests = load_prompt_tests(str(p))
    assert tests[1].expected_answer == ""


# ── Document loader ────────────────────────────────────────────

def test_load_document_returns_content(tmp_path):
    doc = tmp_path / "leave.md"
    doc.write_text("Employees get 25 days annual leave.")
    content = load_document(str(doc))
    assert "25 days" in content


# ── Config loader ──────────────────────────────────────────────

def test_load_test_suite_config(tmp_path):
    p = tmp_path / "test_suite.yaml"
    p.write_text(TEST_SUITE_YAML)
    config = load_test_suite(str(p))
    assert isinstance(config, TestSuiteConfig)
    assert config.provider == "gemini"
    assert len(config.tests) == 1


def test_load_evaluation_config(tmp_path):
    p = tmp_path / "evaluation.yaml"
    p.write_text(EVAL_YAML)
    config = load_evaluation_config(str(p))
    assert isinstance(config, EvaluationConfig)
    assert config.evaluators == ["hallucination", "relevance"]
