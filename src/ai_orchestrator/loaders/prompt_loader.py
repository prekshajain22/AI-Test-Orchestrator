from pathlib import Path
import yaml

from ai_orchestrator.models import PromptTestCase


def load_prompt_tests(file_path: str) -> list[PromptTestCase]:
    """Load prompt test cases from a YAML file."""

    with Path(file_path).open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    tests = []

    for item in data["tests"]:
        tests.append(
            PromptTestCase(
                id=item["id"],
                question=item["question"],
                expected_answer=item["expected_answer"],
                source_document=item["source_document"],
            )
        )

    return tests