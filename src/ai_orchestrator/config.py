from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    hf_api_key: str = os.getenv("HF_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "google/flan-t5-base")
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "256"))


settings = Settings()