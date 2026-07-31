from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


def _parse_list(env_var: str, default: str) -> tuple[str, ...]:
    """Parse a comma-separated environment variable into a tuple of strings."""
    raw = os.getenv(env_var, default)
    return tuple(s.strip() for s in raw.split(",") if s.strip())


@dataclass(frozen=True)
class Settings:
    """
    Single source of truth for all runtime configuration.

    Every setting lives in ``.env`` (or the real environment).  Nothing
    should be duplicated in YAML files — YAML is only used for data that
    is genuinely run-specific (e.g. the list of retriever combinations in
    config/comparison.yaml) or for inline comments that document the data
    schema (e.g. sample_data/prompts/*.yaml).

    .env variables
    ──────────────
    # LLM provider ─────────────────────────────────────────────────────
    PROVIDER=gemini              # gemini | huggingface
    GEMINI_API_KEY=              # required when PROVIDER=gemini
    HF_API_KEY=                  # required when PROVIDER=huggingface
    MODEL_NAME=gemini-flash-lite-latest

    # Inference parameters ───────────────────────────────────────────────
    TEMPERATURE=0
    MAX_TOKENS=256

    # Test execution ─────────────────────────────────────────────────────
    TEST_SUITES=sample_data/prompts/hr_questions.yaml   # comma-separated
    EVALUATORS=hallucination,relevance,faithfulness      # comma-separated
    REPORTS=json,html,pdf                                # comma-separated

    # RAG retrieval ──────────────────────────────────────────────────────
    RAG_ENABLED=false            # true | false
    RAG_TOP_K=3
    RAG_RETRIEVER=tfidf          # tfidf | bm25 | sentence_transformer | faiss | chroma
    """

    # ── LLM provider ────────────────────────────────────────────────────
    provider: str = os.getenv("PROVIDER", "gemini")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    hf_api_key: str = os.getenv("HF_API_KEY", "")
    model_name: str = os.getenv("MODEL_NAME", "google/flan-t5-base")

    # ── Inference parameters ─────────────────────────────────────────────
    temperature: float = float(os.getenv("TEMPERATURE", "0"))
    max_tokens: int = int(os.getenv("MAX_TOKENS", "256"))

    # ── Test execution ───────────────────────────────────────────────────
    test_suites: tuple[str, ...] = _parse_list(
        "TEST_SUITES", "sample_data/prompts/hr_questions.yaml"
    )
    evaluators: tuple[str, ...] = _parse_list(
        "EVALUATORS", "hallucination,relevance,faithfulness"
    )
    reports: tuple[str, ...] = _parse_list("REPORTS", "json,html,pdf")

    # ── RAG retrieval ────────────────────────────────────────────────────
    rag_enabled: bool = os.getenv("RAG_ENABLED", "false").lower() == "true"
    rag_top_k: int = int(os.getenv("RAG_TOP_K", "3"))
    rag_retriever: str = os.getenv("RAG_RETRIEVER", "tfidf")


settings = Settings()
