from ai_orchestrator.runners.runner import TestRunner


runner = TestRunner(
    "sample_data/prompts/hr_questions.yaml"
)

runner.run()