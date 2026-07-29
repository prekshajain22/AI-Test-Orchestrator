from ai_orchestrator.evaluators import (
    EvaluationEngine,
    FaithfulnessEvaluator,
    HallucinationEvaluator,
    RelevanceEvaluator,
)
from ai_orchestrator.loaders import (
    load_document,
    load_prompt_tests,
)
from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.providers.gemini import GeminiProvider
from ai_orchestrator.reporting import JsonReport


class TestRunner:
    """
    Executes AI test cases and evaluates responses.
    """

    def __init__(self):
        self.provider = GeminiProvider()

        self.engine = EvaluationEngine()
        self.engine.register(HallucinationEvaluator())
        self.engine.register(RelevanceEvaluator())
        self.engine.register(FaithfulnessEvaluator())

        self.reporter = JsonReport()

    def run(self) -> list[TestExecutionResult]:

        tests = load_prompt_tests(
            "sample_data/prompts/hr_questions.yaml"
        )

        execution_results: list[TestExecutionResult] = []

        for test in tests:

            print("=" * 60)
            print(f"Running: {test.id}")

            context = load_document(
                test.source_document
            )

            answer = self.provider.ask(
                test.question,
                context,
            )

            print("\nQuestion:")
            print(test.question)

            print("\nAI Answer:")
            print(answer)

            evaluations = self.engine.evaluate(
                test_id=test.id,
                question=test.question,
                answer=answer,
                context=context,
            )

            print("\nEvaluation Results:")

            for result in evaluations:
                print("-" * 40)
                print(f"Metric: {result.metric}")
                print(f"Score: {result.score}")
                print(
                    f"Status: {'PASSED' if result.passed else 'FAILED'}"
                )
                print(f"Reason: {result.reason}")

            execution_result = TestExecutionResult(
                test_id=test.id,
                question=test.question,
                answer=answer,
                evaluations=evaluations,
            )

            execution_results.append(
                execution_result
            )

        report_path = self.reporter.generate(
            execution_results
        )

        print()
        print("=" * 60)
        print(f"JSON report generated: {report_path}")

        return execution_results