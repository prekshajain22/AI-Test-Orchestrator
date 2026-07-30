import requests

from ai_orchestrator.config.settings import settings
from ai_orchestrator.providers.base import LLMClient
from ai_orchestrator.prompts import render_qa_prompt


class HuggingFaceClient(LLMClient):
    """Client for the Hugging Face Inference API."""

    def __init__(self):
        self.api_key = settings.hf_api_key
        self.model_name = settings.model_name

        self.url = (
            f"https://api-inference.huggingface.co/models/"
            f"{self.model_name}"
        )

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def ask(self, question: str, context: str) -> str:
        """Send a question and context to the model."""

        prompt = render_qa_prompt(question, context)

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature": settings.temperature,
                "max_new_tokens": settings.max_tokens,
                "return_full_text": False,
            },
        }

        response = requests.post(
            self.url,
            headers=self.headers,
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        result = response.json()

        if isinstance(result, list):
            return result[0].get("generated_text", "").strip()

        if isinstance(result, dict):

            if "generated_text" in result:
                return result["generated_text"].strip()

            if "error" in result:
                raise RuntimeError(result["error"])

        raise RuntimeError(f"Unexpected Hugging Face response: {result}")
