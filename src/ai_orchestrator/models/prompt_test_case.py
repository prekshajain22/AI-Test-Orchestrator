from dataclasses import dataclass

@dataclass
class PromptTestCase:
    id: str
    question: str
    expected_answer: str
    source_document: str