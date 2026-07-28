from pathlib import Path

from ai_orchestrator.providers.gemini import GeminiProvider

context = Path(
    "sample_data/documents/sick_leave.md"
).read_text(encoding="utf-8")

client = GeminiProvider()

answer = client.ask(
    "When is a medical certificate required?",
    context,
)

print(answer)