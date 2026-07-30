from google import genai

from ai_orchestrator.providers.base import LLMClient, ProviderRateLimitError
from ai_orchestrator.config.settings import settings
from ai_orchestrator.prompts import render_qa_prompt
from google.genai.errors import ClientError

class GeminiProvider(LLMClient):

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

        self.model = settings.model_name

    def ask(self, question: str, context: str) -> str:

        prompt = render_qa_prompt(question, context)

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            return response.text

        except ClientError as e:
            if e.code == 429:
                raise ProviderRateLimitError(
                    "Gemini API quota exceeded (HTTP 429). "
                    "Please check provider limits."
                ) from e

            raise
