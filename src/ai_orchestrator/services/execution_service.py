import logging

from ai_orchestrator.evaluators.engine import EvaluationEngine
from ai_orchestrator.evaluators import EvaluationFactory
from ai_orchestrator.loaders import load_document, load_prompt_tests
from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.providers.factory import ProviderFactory

logger = logging.getLogger(__name__)


class ExecutionService:
    """
    Executes a full test run end-to-end.

    Receives everything it needs via its constructor instead of reading
    configuration files itself:
      - provider_name: which LLM provider to use (e.g. "gemini")
      - test_suites: any number of prompt test suite YAML file paths
      - evaluators: which evaluators to run against each AI response

    Configuration loading is the caller's responsibility (see
    ai_orchestrator.config.loader.load_execution_config and
    ai_orchestrator.runners.runner.TestRunner), which keeps this service
    decoupled from how/where config is stored, and combinable with any
    number of test suite files.

    Does not know about reporting or how results are displayed.
    """

    def __init__(
        self,
        provider_name: str,
        test_suites: list[str],
        evaluators: list[str],
    ):
        self.provider = ProviderFactory.create(provider_name)
        self.test_paths = test_suites

        self.engine = EvaluationEngine()
        for evaluator in EvaluationFactory.create_all(evaluators):
            self.engine.register(evaluator)

    def execute(self) -> list[TestExecutionResult]:
        """Run all configured test suites."""
        execution_results: list[TestExecutionResult] = []

        for prompts_path in self.test_paths:
            logger.info("Loading test suite: %s", prompts_path)
            execution_results.extend(self._run_suite(prompts_path))

        logger.info("Execution complete. %d test(s) run.", len(execution_results))
        return execution_results

    def _run_suite(self, prompts_path: str) -> list[TestExecutionResult]:
        tests = load_prompt_tests(prompts_path)
        results: list[TestExecutionResult] = []

        for test in tests:
            logger.info("Running test: %s", test.id)

            context = load_document(test.source_document)
            answer = self.provider.ask(test.question, context)

            logger.debug("Question: %s", test.question)
            logger.debug("Answer:   %s", answer)

            evaluations = self.engine.evaluate(
                test_id=test.id,
                question=test.question,
                answer=answer,
                context=context,
            )

            for ev in evaluations:
                status = "PASSED" if ev.passed else "FAILED"
                logger.info(
                    "  [%s] %s  score=%.2f  reason=%s",
                    status, ev.metric, ev.score, ev.reason,
                )

            results.append(
                TestExecutionResult(
                    test_id=test.id,
                    question=test.question,
                    answer=answer,
                    evaluations=evaluations,
                )
            )

        return results
