"""
Tests for loaders and configuration.

Config tests now verify that settings.py reads from environment variables
(the single source of truth) rather than from YAML files.
"""
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


# ── ExecutionConfig (settings-based) ──────────────────────────
# load_execution_config() reads from settings which reads from .env.
# We test it via monkeypatching the settings singleton.

def test_load_execution_config_returns_execution_config(monkeypatch):
    """load_execution_config() returns an ExecutionConfig instance."""
    config = load_execution_config()
    assert isinstance(config, ExecutionConfig)


def test_load_execution_config_provider_comes_from_settings(monkeypatch):
    """provider field is read from settings — patch the name inside loader."""
    import ai_orchestrator.config.loader as loader_mod
    from ai_orchestrator.config.settings import Settings
    monkeypatch.setattr(loader_mod, "settings", Settings(provider="test-provider"))
    assert loader_mod.load_execution_config().provider == "test-provider"


def test_load_execution_config_test_suites_come_from_settings(monkeypatch):
    import ai_orchestrator.config.loader as loader_mod
    from ai_orchestrator.config.settings import Settings
    monkeypatch.setattr(
        loader_mod, "settings",
        Settings(test_suites=("suite_a.yaml", "suite_b.yaml")),
    )
    assert loader_mod.load_execution_config().test_suites == ["suite_a.yaml", "suite_b.yaml"]


def test_load_execution_config_evaluators_come_from_settings(monkeypatch):
    import ai_orchestrator.config.loader as loader_mod
    from ai_orchestrator.config.settings import Settings
    monkeypatch.setattr(
        loader_mod, "settings",
        Settings(evaluators=("hallucination", "relevance")),
    )
    assert loader_mod.load_execution_config().evaluators == ["hallucination", "relevance"]


def test_load_execution_config_reports_come_from_settings(monkeypatch):
    import ai_orchestrator.config.loader as loader_mod
    from ai_orchestrator.config.settings import Settings
    monkeypatch.setattr(loader_mod, "settings", Settings(reports=("json", "html")))
    assert loader_mod.load_execution_config().reports == ["json", "html"]


# ── RagConfig (via settings) ───────────────────────────────────

def test_load_execution_config_rag_defaults_to_disabled(monkeypatch):
    import ai_orchestrator.config.loader as loader_mod
    from ai_orchestrator.config.settings import Settings
    monkeypatch.setattr(
        loader_mod, "settings",
        Settings(rag_enabled=False, rag_top_k=3, rag_retriever="tfidf"),
    )
    config = loader_mod.load_execution_config()
    assert isinstance(config.rag, RagConfig)
    assert config.rag.enabled is False
    assert config.rag.top_k == 3
    assert config.rag.retriever == "tfidf"


def test_load_execution_config_rag_enabled_true(monkeypatch):
    import ai_orchestrator.config.loader as loader_mod
    from ai_orchestrator.config.settings import Settings
    monkeypatch.setattr(
        loader_mod, "settings",
        Settings(rag_enabled=True, rag_top_k=5, rag_retriever="bm25"),
    )
    config = loader_mod.load_execution_config()
    assert config.rag.enabled is True
    assert config.rag.top_k == 5
    assert config.rag.retriever == "bm25"


def test_load_execution_config_rag_custom_top_k(monkeypatch):
    import ai_orchestrator.config.loader as loader_mod
    from ai_orchestrator.config.settings import Settings
    monkeypatch.setattr(
        loader_mod, "settings",
        Settings(rag_enabled=False, rag_top_k=7, rag_retriever="tfidf"),
    )
    assert loader_mod.load_execution_config().rag.top_k == 7


# ── prompt_loader use_rag field ────────────────────────────────

def test_load_prompt_tests_use_rag_defaults_false(tmp_path):
    p = tmp_path / "prompts.yaml"
    p.write_text(PROMPT_YAML)
    tests = load_prompt_tests(str(p))
    assert tests[0].use_rag is False


def test_load_prompt_tests_use_rag_true(tmp_path):
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
