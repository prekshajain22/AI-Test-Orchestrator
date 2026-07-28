# Architecture Decisions

## ADR-001: Provider Abstraction

Decision:

Create a common LLM provider interface.

Reason:

The framework should support multiple LLM providers without changing test execution logic.

Current Providers:

- Gemini

Future Providers:

- OpenAI
- Azure OpenAI
- Hugging Face

Status:

Implemented
