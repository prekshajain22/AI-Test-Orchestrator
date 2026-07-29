from ai_orchestrator.config.settings import settings


def test_settings_loaded():
    """Settings object is importable and has all expected fields."""
    assert isinstance(settings.model_name, str)
    assert isinstance(settings.gemini_api_key, str)
    assert isinstance(settings.hf_api_key, str)
    assert isinstance(settings.temperature, float)
    assert isinstance(settings.max_tokens, int)
