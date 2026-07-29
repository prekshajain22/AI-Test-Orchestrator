import logging

from ai_orchestrator.config.loader import load_evaluation_config, load_test_suite
from ai_orchestrator.evaluators.engine import EvaluationEngine
from ai_orchestrator.evaluators.factory import EvaluationFactory
from ai_orchestrator.loaders import load_document, load_prompt_tests
from ai_orchestrator.models import TestExecutionResult
from ai_orchestrator.providers.factory import ProviderFactory

logger = logging.getLogger(__name__)


class ExecutionService:
    """
    Executes a full test suite end-to-end.

    Configuration is read from:
      - config/test_suite.yaml  (provider, list of prompt files)
      - config/evaluation.yaml  (which evaluators to run)

    Does not know about reporting or how results are displayed.
    """

    TEST_SUITE_CONFIG = "config/test_suite.yaml"
    EVALUATION_CONFIG = "config/evaluation.yaml"

    def __init__(self):
        suite_config = load_test_suite(self.TEST_SUITE_CONFIG)
        eval_config = load_evaluation_config(self.EVALUATION_CONFIG)

        self.provider = ProviderFactory.create(suite_config.provider)
        self.test_paths = suite_config.tests

        self.engine = EvaluationEngine()
        for evaluator in EvaluationFactory.create_all(eval_config.evaluators):
            self.engine.register(evaluator)

    def execute(self) -> list[TestExecutionResult]:
        """Run all test suites defined in config/test_suite.yaml."""
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
