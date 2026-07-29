from dataclasses import dataclass, field


@dataclass
class ExecutionMetadata:
    """
    Metadata describing the context in which a test execution run happened.

    Making this a first-class part of the report output means every
    generated report (JSON/HTML/PDF) can be traced back to exactly:
      - which run it was (execution_id)
      - when it ran (timestamp)
      - which LLM provider/model answered the questions
      - what temperature was used
      - which test suite(s) were executed
    """

    execution_id: str
    timestamp: str
    provider: str = "unknown"
    model: str = "unknown"
    temperature: float = 0.0
    test_suite: list[str] = field(default_factory=list)
