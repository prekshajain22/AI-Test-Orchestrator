import textwrap

import pytest

from ai_orchestrator.loaders.prompt_loader import load_prompt_tests
from ai_orchestrator.loaders.document_loader import load_document
from ai_orchestrator.config.loader import (
    load_execution_config,
    ExecutionConfig,
    RagConfig,
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

EXECUTION_YAML = textwrap.dedent("""\
    provider:
      name: gemini
    test_suites:
      - sample_data/prompts/hr_questions.yaml
    evaluators:
      - hallucination
      - relevance
    reports:
      - json
      - html
""")

EXECUTION_YAML_STRING_PROVIDER = textwrap.dedent("""\
    provider: gemini
    test_suites:
      - sample_data/prompts/hr_questions.yaml
    evaluators:
      - hallucination
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

def test_load_execution_config(tmp_path):
    p = tmp_path / "execution.yaml"
    p.write_text(EXECUTION_YAML)
    config = load_execution_config(str(p))
    assert isinstance(config, ExecutionConfig)
    assert config.provider == "gemini"
    assert config.test_suites == ["sample_data/prompts/hr_questions.yaml"]
    assert config.evaluators == ["hallucination", "relevance"]
    assert config.reports == ["json", "html"]


def test_load_execution_config_supports_multiple_test_suites(tmp_path):
    p = tmp_path / "execution.yaml"
    p.write_text(
        textwrap.dedent("""\
            provider:
              name: gemini
            test_suites:
              - suite_one.yaml
              - suite_two.yaml
              - suite_three.yaml
            evaluators:
              - hallucination
        """)
    )
    config = load_execution_config(str(p))
    assert len(config.test_suites) == 3


def test_load_execution_config_accepts_string_provider(tmp_path):
    p = tmp_path / "execution.yaml"
    p.write_text(EXECUTION_YAML_STRING_PROVIDER)
    config = load_execution_config(str(p))
    assert config.provider == "gemini"
    assert config.evaluators == ["hallucination"]


# ── RagConfig parsing ─────────────────────────────────────────────────────────

def test_load_execution_config_rag_defaults_to_disabled(tmp_path):
    """When the YAML has no 'rag' key, RagConfig should default to disabled."""
    p = tmp_path / "execution.yaml"
    p.write_text(EXECUTION_YAML)
    config = load_execution_config(str(p))
    assert isinstance(config.rag, RagConfig)
    assert config.rag.enabled is False
    assert config.rag.top_k == 3
    assert config.rag.retriever == "tfidf"


def test_load_execution_config_rag_enabled(tmp_path):
    """A rag: enabled: true block should set RagConfig.enabled to True."""
    yaml_text = textwrap.dedent("""\
        provider: gemini
        test_suites:
          - suite.yaml
        evaluators:
          - hallucination
        rag:
          enabled: true
          top_k: 5
          retriever: tfidf
    """)
    p = tmp_path / "execution.yaml"
    p.write_text(yaml_text)
    config = load_execution_config(str(p))
    assert config.rag.enabled is True
    assert config.rag.top_k == 5
    assert config.rag.retriever == "tfidf"


def test_load_execution_config_rag_partial_overrides(tmp_path):
    """Unspecified rag keys fall back to their defaults."""
    yaml_text = textwrap.dedent("""\
        provider: gemini
        test_suites: []
        evaluators: []
        rag:
          top_k: 7
    """)
    p = tmp_path / "execution.yaml"
    p.write_text(yaml_text)
    config = load_execution_config(str(p))
    assert config.rag.enabled is False   # default
    assert config.rag.top_k == 7
    assert config.rag.retriever == "tfidf"  # default


# ── prompt_loader use_rag field ───────────────────────────────────────────────

def test_load_prompt_tests_use_rag_defaults_false(tmp_path):
    """Test cases without use_rag should default to False."""
    p = tmp_path / "prompts.yaml"
    p.write_text(PROMPT_YAML)
    tests = load_prompt_tests(str(p))
    assert tests[0].use_rag is False


def test_load_prompt_tests_use_rag_true(tmp_path):
    """Test cases with use_rag: true should set the flag."""
    yaml_text = textwrap.dedent("""\
        tests:
          - id: t_rag
            question: "What is the policy?"
            source_document: "docs/policy.md"
            use_rag: true
    """)
    p = tmp_path / "prompts.yaml"
    p.write_text(yaml_text)
    tests = load_prompt_tests(str(p))
    assert tests[0].use_rag is True
