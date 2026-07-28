from ai_orchestrator.evaluators import HallucinationEvaluator


evaluator = HallucinationEvaluator()


result = evaluator.evaluate(
    test_id="remote_days_test",
    question="How many days can employees work remotely?",
    answer="Employees can work remotely 3 days per week.",
    context="The policy does not define a fixed number of remote days. Employees should agree a schedule with their manager."
)


print(result)