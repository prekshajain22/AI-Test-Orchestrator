# Architecture

## Overview

AI Test Orchestrator is a modular framework for testing LLM-based applications: pluggable LLM providers, pluggable retrievers, pluggable evaluators, and a config-driven runner that ties them together, producing structured reports.

The goal is to apply the same discipline traditional QA applies to deterministic systems — repeatable test cases, ground-truth comparison, pass/fail reporting — to a domain where "correct" is fuzzier and has to be scored rather than asserted.

---

## Configuration: single source of truth

**All runtime configuration lives in `.env`** (loaded into `Settings` via `config/settings.py`). No YAML file is read for configuration — YAML is only used for data definitions (prompt test cases, comparison retriever lists).

```
.env
 └── Settings (config/settings.py)
       ├── provider, gemini_api_key, hf_api_key, model_name
       ├── temperature, max_tokens
       ├── test_suites, evaluators, reports         ← what to run
       └── rag_enabled, rag_top_k, rag_retriever    ← RAG settings
```

`load_execution_config()` (config/loader.py) reads `Settings` and returns an `ExecutionConfig` dataclass — no file path argument needed.

---

## Pipeline

```
Settings (.env)
     │
     ▼
TestRunner
     │  builds ExecutionService + ReportManager from Settings
     │
     ▼
ExecutionService
     │
     ├─ load_prompt_tests()    → list[PromptTestCase]
     │
     ├─ for each test:
     │    ├─ load_document()           → full document text
     │    ├─ [if RAG] chunk_document() → list[Chunk]
     │    │           retriever.retrieve() → list[RetrievedChunk]
     │    │           → RetrievalMetrics (hit@k, MRR, coverage, avg_sim)
     │    ├─ provider.ask()            → answer string
     │    └─ engine.evaluate()         → list[EvaluationResult]
     │
     ▼
ReportManager
     └─ JSON / HTML / PDF reports  (ExecutionSummary + per-test detail)
```

### Comparison runs

```
config/comparison.yaml   (retriever strategies only)
     │
     ▼
ComparisonRunner
     └─ runs ExecutionService once per retriever strategy
     └─ ComparisonReport → reports/comparison/*.html
                           (side-by-side metric table, ★ BEST badge)
```

### Dashboard

```
reports/json/execution_*.json   (accumulated across all runs)
     │
     ▼
Dashboard
     └─ reports/dashboard.html
        (KPI cards, execution history, provider comparison)
```

---

## Providers

`LLMClient` is an abstract base class: `ask(question, context) -> str`.

| Provider     | Class               | Notes                  |
| ------------ | ------------------- | ---------------------- |
| Gemini       | `GeminiProvider`    | via `google-genai` SDK |
| Hugging Face | `HuggingFaceClient` | via Inference API      |

Set `PROVIDER=gemini` (or `huggingface`) in `.env`. New providers plug in by implementing `LLMClient` and registering in `ProviderRegistry`.

---

## Evaluators

`BaseEvaluator.evaluate()` returns `list[EvaluationResult]` — a list so that a single evaluator (e.g. `LlmJudgeEvaluator`) can return multiple scored dimensions from one API call. `EvaluationEngine` flattens results from all registered evaluators.

| Evaluator                | Type           | Dimensions                                                                                        | Pass threshold |
| ------------------------ | -------------- | ------------------------------------------------------------------------------------------------- | -------------- |
| `HallucinationEvaluator` | Heuristic      | `hallucination`                                                                                   | ≥ 0.7          |
| `RelevanceEvaluator`     | Heuristic      | `relevance`                                                                                       | ≥ 0.5          |
| `FaithfulnessEvaluator`  | Heuristic      | `faithfulness`                                                                                    | ≥ 0.7          |
| `LlmJudgeEvaluator`      | LLM-as-a-Judge | `llm_judge_correctness` `llm_judge_completeness` `llm_judge_groundedness` `llm_judge_helpfulness` | ≥ 0.7 each     |

**Evaluation philosophy:** The three heuristic evaluators are intentionally naive, hand-rolled lexical-overlap implementations — built from scratch rather than using DeepEval/Ragas specifically to understand what these metrics actually measure. They have known weaknesses (they penalise correct paraphrasing; they can miss hallucinations that reuse source vocabulary). They are planned to be replaced with DeepEval/Ragas internals behind the same `BaseEvaluator` interface, so no architecture change is required — only what's plugged in.

`LlmJudgeEvaluator` makes one structured JSON API call per test and returns four scores. Graceful degradation: provider failure → score=0, run never crashes.

---

## RAG — Retrieval-Augmented Generation

When `RAG_ENABLED=true` (or a test case sets `use_rag: true`), `ExecutionService` replaces the full-document context with the top-k most relevant chunks retrieved from the document.

```
chunk_document(text)          →  list[Chunk]   (heading-first, paragraph fallback)
retriever.retrieve(q, chunks) →  list[RetrievedChunk]  (ranked by score)
"\n\n".join(rc.chunk.text)    →  context sent to LLM
```

`BaseRetriever` hierarchy:

```
BaseRetriever
     ├── TfidfRetriever          pure Python, default
     ├── BM25Retriever           pure Python, Okapi BM25
     ├── SentenceTransformerRetriever   sentence-transformers (optional)
     ├── FaissRetriever          faiss-cpu + sentence-transformers (optional)
     └── ChromaRetriever         chromadb (optional)
```

Heavy-dependency retrievers are lazily imported — the app starts fine without those packages; the `ImportError` only surfaces if you actually try to use them.

**Per-test retrieval metrics** (`RetrievalMetrics`) are stored on `TestExecutionResult.retrieval_metrics` and logged as a `[RETRIEVAL]` line:

| Metric               | Definition                                            |
| -------------------- | ----------------------------------------------------- |
| `chunks_retrieved`   | Count of chunks returned (≤ top_k)                    |
| `chunk_scores`       | Relevance scores, highest first                       |
| `average_similarity` | Mean chunk score                                      |
| `hit_at_k`           | ≥ 1 chunk contains ≥ 50% of expected-answer tokens    |
| `mrr`                | 1/rank of first relevant chunk (Mean Reciprocal Rank) |
| `context_coverage`   | % of expected-answer vocabulary in retrieved context  |

---

## Reporting

| Report     | Class              | Output                                 |
| ---------- | ------------------ | -------------------------------------- |
| JSON       | `JsonReport`       | `reports/json/execution_*.json`        |
| HTML       | `HtmlReport`       | `reports/html/execution_*.html`        |
| PDF        | `PdfReport`        | `reports/pdf/execution_*.pdf`          |
| Comparison | `ComparisonReport` | `reports/comparison/comparison_*.html` |
| Dashboard  | `Dashboard`        | `reports/dashboard.html`               |

All reports are tagged with `ExecutionMetadata` (execution ID, timestamp, provider, model, temperature, test suite) so every report is fully traceable.

---

## Key design decisions

**`.env` as single source of truth.** Provider, test suites, evaluators, reports, and RAG settings are configured in exactly one place (`.env`). No config duplication across YAML files. YAML is reserved for data definitions that are inherently list-structured (prompt test cases, comparison retriever strategies).

**Provider abstraction.** A common `LLMClient` interface decouples execution from the specific provider so Gemini and Hugging Face are interchangeable. `LlmJudgeEvaluator` always uses the same provider as the main answer-generation step (`settings.provider`), never a hard-coded default.

**Evaluator abstraction.** `BaseEvaluator.evaluate()` returns a list so that multi-dimension evaluators (LLM judge) sit alongside single-dimension heuristics behind the same interface. `EvaluationEngine.evaluate()` returns a flat `list[EvaluationResult]` regardless.

**Retriever abstraction.** `BaseRetriever` follows the same Strategy pattern as providers and evaluators — swap implementations without changing `ExecutionService`.

**Constructor injection.** `ExecutionService` receives provider, evaluators, and RAG config via constructor arguments rather than reading config files itself. This keeps it decoupled, testable, and combinable (e.g. `ComparisonRunner` constructs one `ExecutionService` per retriever strategy).

---

## Known limitations

- Heuristic evaluators are lexical-overlap, not semantic/embedding-based.
- Execution is sequential — no concurrency, no LLM response caching.
- No retry/backoff on provider network calls.

---

## Roadmap

**v0.1 Foundation** — ✅ Repo setup, provider abstraction, initial docs.

**v0.2 LLM Test Execution** — ✅ YAML test cases, document loading, Gemini integration, test runner.

**v0.3 Evaluation Engine** — ✅ Hallucination, relevance, faithfulness evaluators; JSON/HTML/PDF reporting.

**v0.4 RAG + Retrieval Metrics** — ✅ Chunker, TF-IDF/BM25/ST/FAISS/Chroma retrievers, retrieval metrics (hit@k, MRR, coverage), RAG behind config flag, comparison runner, dashboard.

**v0.5 LLM-as-a-Judge + Single Config Source** — ✅ `LlmJudgeEvaluator` (4 dimensions), `.env` as single source of truth, comparison report HTML, 193 tests.

**v0.6 Real Evaluation Metrics** — Planned: replace heuristic evaluators with DeepEval/Ragas behind the same `BaseEvaluator` interface.

**v0.7 Multi-Provider + Resilience** — Planned: OpenAI/Azure OpenAI providers, retry/backoff, response caching.

**v0.8 CI/CD** — ✅ GitHub Actions workflow (`tests.yml`) runs all 193 tests on every push and pull request to `main`/`trunk`. Pip dependency caching included. No real API key required — all tests mock the LLM provider.

**v1.0 Enterprise AI Quality Framework** — Future: concurrent execution, cost tracking, per-chunk attribution in reports.
