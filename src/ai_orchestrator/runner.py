from ai_orchestrator.loaders import load_prompt_tests

tests = load_prompt_tests("sample_data/prompts/hr_questions.yaml")

for test in tests:
    print("=" * 60)
    print(f"ID: {test.id}")
    print(f"Question: {test.question}")
    print(f"Expected: {test.expected_answer}")
    print(f"Source: {test.source_document}")