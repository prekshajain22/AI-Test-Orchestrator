from ai_orchestrator.loaders import load_prompt_tests
from ai_orchestrator.loaders.document_loader import load_document
from ai_orchestrator.providers.gemini import GeminiProvider


class TestRunner:

    def __init__(self, test_file: str):

        self.test_file = test_file
        self.provider = GeminiProvider()


    def load_tests(self):

        return load_prompt_tests(
            self.test_file
        )


    def run(self):

        tests = self.load_tests()

        for test in tests:

            print("=" * 60)
            print(f"Running: {test.id}")

            context = load_document(
                test.source_document
            )


            answer = self.provider.ask(
                test.question,
                context
            )


            print("\nQuestion:")
            print(test.question)

            print("\nExpected:")
            print(test.expected_answer)

            print("\nAI Answer:")
            print(answer)