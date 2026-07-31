from ai_orchestrator.config.settings import settings


def test_settings_loaded():
    """Settings object is importable and has all expected fields."""
    assert isinstance(settings.model_name, str)
    assert isinstance(settings.gemini_api_key, str)
    assert isinstance(settings.hf_api_key, str)
    assert isinstance(settings.temperature, float)
    assert isinstance(settings.max_tokens, int)


def test_settings_provider_is_string():
    """PROVIDER from .env is exposed as settings.provider."""
    assert isinstance(settings.provider, str)
    assert len(settings.provider) > 0


def test_settings_test_suites_is_tuple_of_strings():
    """TEST_SUITES from .env is exposed as a tuple of non-empty strings."""
    assert isinstance(settings.test_suites, tuple)
    assert len(settings.test_suites) > 0
    assert all(isinstance(s, str) and s for s in settings.test_suites)


def test_settings_evaluators_is_tuple_of_strings():
    """EVALUATORS from .env is exposed as a tuple of non-empty strings."""
    assert isinstance(settings.evaluators, tuple)
    assert len(settings.evaluators) > 0
    assert all(isinstance(e, str) and e for e in settings.evaluators)


def test_settings_reports_is_tuple_of_strings():
    """REPORTS from .env is exposed as a tuple of non-empty strings."""
    assert isinstance(settings.reports, tuple)
    assert len(settings.reports) > 0
    assert all(isinstance(r, str) and r for r in settings.reports)


def test_settings_rag_enabled_is_bool():
    assert isinstance(settings.rag_enabled, bool)


def test_settings_rag_top_k_is_positive_int():
    assert isinstance(settings.rag_top_k, int)
    assert settings.rag_top_k > 0


def test_settings_rag_retriever_is_string():
    assert isinstance(settings.rag_retriever, str)
    assert len(settings.rag_retriever) > 0
