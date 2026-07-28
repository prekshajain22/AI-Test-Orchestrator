from ai_orchestrator.config.settings import settings


def test_settings_loaded():
    assert settings.model_name == "google/flan-t5-base"