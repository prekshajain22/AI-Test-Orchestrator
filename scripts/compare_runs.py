"""
Comparison run script.

Executes all (provider, retriever) configurations defined in
config/comparison.yaml and generates a side-by-side HTML comparison report.

Usage:
    python scripts/compare_runs.py
"""
import logging

from ai_orchestrator.config.comparison_loader import load_comparison_config
from ai_orchestrator.runners.comparison_runner import ComparisonRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

config = load_comparison_config("config/comparison.yaml")
runner = ComparisonRunner(config)
run_results = runner.run()

print()
print("=" * 60)
print(f"Comparison complete. {len(run_results)} run(s) executed.")
for rr in run_results:
    total = len(rr.results)
    passed = sum(1 for r in rr.results if r.passed)
    print(f"  {rr.run_config.name:30s}  {passed}/{total} passed")
