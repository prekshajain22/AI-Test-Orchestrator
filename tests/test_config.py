from ai_orchestrator.config import settings


def test_settings_loaded():
    assert settings.model_name == "google/flan-t5-base"