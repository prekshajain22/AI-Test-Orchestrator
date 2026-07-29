from ai_orchestrator.runners.runner import TestRunner

runner = TestRunner()

results = runner.run()

print(f"Total executions: {len(results)}")