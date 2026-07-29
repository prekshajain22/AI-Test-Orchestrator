import pytest

from ai_orchestrator.evaluators import (
    EvaluationEngine,
    EvaluationFactory,
    HallucinationEvaluator,
    RelevanceEvaluator,
    FaithfulnessEvaluator,
)
from ai_orchestrator.models import EvaluationResult


# ── EvaluationFactory ──────────────────────────────────────────

def test_factory_creates_hallucination_evaluator():
    assert isinstance(EvaluationFactory.create("hallucination"), HallucinationEvaluator)


def test_factory_creates_relevance_evaluator():
    assert isinstance(EvaluationFactory.create("relevance"), RelevanceEvaluator)


def test_factory_creates_faithfulness_evaluator():
    assert isinstance(EvaluationFactory.create("faithfulness"), FaithfulnessEvaluator)


def test_factory_create_all_returns_correct_count():
    evaluators = EvaluationFactory.create_all(["hallucination", "relevance", "faithfulness"])
    assert len(evaluators) == 3


def test_factory_raises_for_unknown_evaluator():
    with pytest.raises(ValueError, match="Unknown evaluator"):
        EvaluationFactory.create("unknown_metric")


# ── EvaluationEngine ───────────────────────────────────────────

def test_engine_returns_one_result_per_registered_evaluator():
    engine = EvaluationEngine()
    engine.register(HallucinationEvaluator())
    engine.register(RelevanceEvaluator())

    results = engine.evaluate(
        test_id="t1",
        question="What is the leave policy?",
        answer="Employees get 25 days annual leave.",
        context="Employees are entitled to 25 days of annual leave per year.",
    )
    assert len(results) == 2


def test_engine_results_are_evaluation_result_instances():
    engine = EvaluationEngine()
    engine.register(HallucinationEvaluator())

    results = engine.evaluate(test_id="t1", question="Q", answer="A", context="C")
    assert all(isinstance(r, EvaluationResult) for r in results)


def test_engine_result_score_in_valid_range():
    engine = EvaluationEngine()
    engine.register(HallucinationEvaluator())

    results = engine.evaluate(
        test_id="t1", question="Q", answer="The answer is X.", context="The answer is X."
    )
    for result in results:
        assert 0.0 <= result.score <= 1.0


# ── HallucinationEvaluator ─────────────────────────────────────

def test_hallucination_passes_when_answer_matches_context():
    ev = HallucinationEvaluator()
    result = ev.evaluate(
        test_id="t1",
        question="What is the policy?",
        answer="Employees get 25 days annual leave per year.",
        context="Employees are entitled to 25 days of annual leave per year.",
    )
    assert result.score > 0.5
    assert result.metric == "hallucination"


def test_hallucination_fails_when_answer_diverges_from_context():
    ev = HallucinationEvaluator()
    result = ev.evaluate(
        test_id="t1",
        question="What is the policy?",
        answer="Employees get unlimited paid vacation and free cars.",
        context="Employees work standard 9-5 hours.",
    )
    assert result.score < 0.7
