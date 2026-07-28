from google import genai

from ai_orchestrator.providers.base import LLMClient
from ai_orchestrator.config.settings import settings
from google.genai.errors import ClientError

class GeminiProvider(LLMClient):

    def __init__(self):
        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )

        self.model = settings.model_name

    def ask(self, question: str, context: str) -> str:

        prompt = f"""
        Answer the question using only the provided context.

        Context:
        {context}

        Question:
        {question}
        """

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            return response.text

        except ClientError as e:
            if e.code == 429:
                return "LLM quota exceeded. Please check provider limits."

            raise