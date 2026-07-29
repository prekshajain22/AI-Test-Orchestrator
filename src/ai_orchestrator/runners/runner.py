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
from ai_orchestrator.reporting import (
    ExecutionSummaryBuilder,
    HtmlReport,
    JsonReport,
    PdfReport,
)


class TestRunner:
    """
    Executes AI test cases, evaluates responses, and generates reports.
    """

    def __init__(self):
        self.provider = GeminiProvider()

        self.engine = EvaluationEngine()
        self.engine.register(HallucinationEvaluator())
        self.engine.register(RelevanceEvaluator())
        self.engine.register(FaithfulnessEvaluator())

        self.json_report = JsonReport()
        self.html_report = HtmlReport()
        self.pdf_report = PdfReport()

    def run(self) -> list[TestExecutionResult]:

        tests = load_prompt_tests(
            "sample_data/prompts/hr_questions.yaml"
        )

        execution_results: list[TestExecutionResult] = []

        for test in tests:

            print("=" * 60)
            print(f"Running: {test.id}")

            context = load_document(test.source_document)

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

            execution_results.append(execution_result)

        summary = ExecutionSummaryBuilder.build(execution_results)

        json_path = self.json_report.generate(summary, execution_results)
        html_path = self.html_report.generate(summary, execution_results)

        print()
        print("=" * 60)
        print(f"JSON report: {json_path}")
        print(f"HTML report: {html_path}")

        try:
            pdf_path = self.pdf_report.generate(summary, execution_results)
            print(f"PDF  report: {pdf_path}")
        except ImportError as e:
            print(f"PDF  report: skipped — {e}")

        return execution_results
