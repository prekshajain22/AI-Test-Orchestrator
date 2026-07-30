# Architecture

## Overview

AI Test Orchestrator is a modular framework for testing LLM-based applications: pluggable LLM providers, pluggable evaluators, and a config-driven runner that ties them together, producing structured reports.

The goal is to apply the same discipline traditional QA applies to deterministic systems — repeatable test cases, ground-truth comparison, pass/fail reporting — to a domain where "correct" is fuzzier and has to be scored rather than asserted.

## Current pipeline (implemented)

```
config/execution.yaml
        |
        v
TestRunner  ──loads config, builds services, triggers run──
        |
        v
ExecutionService
        |
        ├─ loads test cases (YAML) ──> PromptTestCase
        ├─ loads source document ───> ground-truth context
        ├─ calls LLM provider ──────> AIResponse
        └─ runs EvaluationEngine ───> EvaluationResult[]
        |
        v
ReportManager ──builds ExecutionSummary, writes JSON/HTML/PDF──
```

### Providers

`LLMClient` is an abstract base class with one method, `ask(question, context) -> str`. Two implementations exist today: `GeminiProvider` and `HuggingFaceClient`. New providers (OpenAI, Azure OpenAI) plug in by implementing the same interface and registering in `ProviderRegistry`.

### Evaluators

`BaseEvaluator` is an abstract base class with one method, `evaluate(test_id, question, answer, context) -> EvaluationResult`. Three implementations exist today — see "Evaluation Philosophy" below for what they actually do and why.

### Reporting

`ReportManager` coordinates `JsonReport`, `HtmlReport`, and `PdfReport`, all built from a single `ExecutionSummary` (aggregated pass/fail + per-metric stats) plus the raw `TestExecutionResult` list. Every report is tagged with `ExecutionMetadata` (execution ID, timestamp, provider, model, temperature, test suite) so any report can be traced back to exactly how it was produced.

## Evaluation philosophy

Three quality dimensions are evaluated today, each scored 0.0–1.0 against a pass threshold:

- **Hallucination** — does the answer contain claims not supported by the source document? Currently measured as: strip stopwords from both answer and context, take the set difference, score by what fraction of answer-words are unsupported.
- **Relevance** — does the answer address the question asked? Currently measured as keyword overlap (with a small hand-built synonym table) between question and answer.
- **Faithfulness** — is each sentence in the answer traceable back to a sentence in the context? Currently measured as per-sentence keyword overlap, averaged across the answer.

**These are intentionally naive, hand-rolled implementations.** This is my first AI engineering project, and I built these from scratch — rather than starting with DeepEval or Ragas — specifically to understand what these metrics actually measure before relying on a library to compute them for me. They work as a first pass but have known weaknesses: they penalize correct paraphrasing (different words, same meaning) and can miss hallucinations that happen to reuse source vocabulary.

**Planned:** replace the internals of these three evaluators with DeepEval and/or Ragas implementations behind the same `BaseEvaluator` interface, so the pluggable architecture doesn't need to change — only what's plugged in.

Two additional quality dimensions are defined conceptually but have no evaluator yet:

- **Completeness** — does the answer cover all relevant points from the source?
- **Safety** — does the answer avoid leaking sensitive information it shouldn't (e.g. confidential HR/medical data)?

## Decisions

**Provider abstraction.** A common `LLMClient` interface was built from the start so the framework can support multiple LLM providers without changing test execution logic. Current providers: Gemini, Hugging Face. Planned: OpenAI, Azure OpenAI.

**Evaluator abstraction.** Same reasoning as providers — evaluation logic is decoupled from execution logic via `BaseEvaluator`, so the hand-rolled heuristics can be swapped for DeepEval/Ragas without touching `ExecutionService` or the reporting layer.

**Config-driven execution.** A single `config/execution.yaml` controls provider, test suites, evaluators, and report formats, injected into services via constructor arguments rather than read ad hoc throughout the codebase.

## Known limitations

Being upfront about the current gaps rather than letting the docs overstate things:

- Evaluators are lexical-overlap heuristics, not semantic/embedding-based (see above).
- Execution is fully sequential — no concurrency, no caching of LLM responses across runs.
- No CI configured yet — tests exist (~30 in `tests/`) but nothing runs them automatically on push.
- No retry/backoff on provider network calls.

## Roadmap

**v0.1 Foundation** — Done: repo setup, provider abstraction, initial docs.

**v0.2 LLM Test Execution** — Done: YAML test cases, document loading, Gemini integration, test runner.

**v0.3 Evaluation Engine** — Done (hand-rolled version): hallucination, relevance, faithfulness evaluators; reporting (JSON/HTML/PDF).

**v0.4 Real Evaluation Metrics** — In progress: replace hand-rolled evaluators with DeepEval/Ragas; add completeness and safety evaluators.

**v0.5 Multi-Provider + Resilience** — Planned: OpenAI/Azure OpenAI providers, retry/backoff, response caching.

**v0.6 CI/CD** — Planned: GitHub Actions running the test suite on every push; automated regression runs on a schedule.

**v1.0 Enterprise AI Quality Framework** — Future: concurrent execution, dashboard/trend visualization, cost tracking.
