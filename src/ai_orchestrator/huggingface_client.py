import requests

from ai_orchestrator.config import settings
from ai_orchestrator.llm_client import LLMClient


class HuggingFaceClient(LLMClient):
    """Hugging Face Inference API client."""

    def __init__(self):
        self.api_key = settings.hf_api_key
        self.model_name = settings.model_name

        self.url = (
            f"https://api-inference.huggingface.co/models/"
            f"{self.model_name}"
        )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}"
        }