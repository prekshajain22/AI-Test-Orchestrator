# AI Test Orchestrator

A learning project building an AI Quality Engineering framework from first principles: testing LLM-based applications the way a QA engineer would test any other system — with structured test cases, ground-truth source documents, automated evaluation, and repeatable reports.

This is my first hands-on AI engineering project, built and documented as I go. See [`docs/learning-journal.md`](docs/learning-journal.md) for the day-by-day build log, and [`docs/architecture.md`](docs/architecture.md) for the technical design.

---

## What this does today

Given a set of questions, a source document (ground truth), and an LLM provider:

1. Loads test cases from YAML (`sample_data/prompts/`)
2. Optionally chunks documents and retrieves the most relevant passages (RAG)
3. Sends the question + context to an LLM provider (Gemini or Hugging Face)
4. Scores the response against up to four evaluators (heuristic + LLM-as-a-Judge)
5. Generates JSON, HTML, and PDF reports with pass/fail rates and per-metric breakdowns
6. Optionally runs multiple retriever strategies side-by-side and generates a comparison report

---

## Setup

```bash
git clone https://github.com/prekshajain22/AI-Test-Orchestrator.git
cd AI-Test-Orchestrator
pip install -r requirements.txt
cp .env.example .env   # fill in GEMINI_API_KEY and configure settings
python scripts/run_tests.py
```

![Tests](https://github.com/prekshajain22/AI-Test-Orchestrator/actions/workflows/tests.yml/badge.svg)

Reports are written to `reports/json/`, `reports/html/`, and `reports/pdf/`.

---

## Configuration — single source of truth: `.env`

**All configuration lives in `.env`** — no need to touch any YAML file for normal runs.

```env
# ── LLM provider ──────────────────────────────────────────────
PROVIDER=gemini              # gemini | huggingface
GEMINI_API_KEY=<your-key>
MODEL_NAME=models/gemini-flash-lite-latest

# ── Test execution ─────────────────────────────────────────────
TEST_SUITES=sample_data/prompts/hr_questions.yaml
EVALUATORS=hallucination,relevance,faithfulness
REPORTS=json,html,pdf

# ── RAG retrieval ──────────────────────────────────────────────
RAG_ENABLED=false            # true | false
RAG_TOP_K=3
RAG_RETRIEVER=tfidf          # tfidf | bm25 | sentence_transformer | faiss | chroma
```

See `.env.example` for the full reference with all available values.

---

## Evaluators

| Name            | Type           | What it measures                                                                            |
| --------------- | -------------- | ------------------------------------------------------------------------------------------- |
| `hallucination` | Heuristic      | Set-difference of answer vs context keywords                                                |
| `relevance`     | Heuristic      | Keyword overlap between question and answer                                                 |
| `faithfulness`  | Heuristic      | Per-sentence faithfulness to source context                                                 |
| `llm_judge`     | LLM-as-a-Judge | Correctness, Completeness, Groundedness, Helpfulness — one LLM call, four scored dimensions |

Add evaluators to `.env`:

```env
EVALUATORS=hallucination,relevance,faithfulness,llm_judge
```

---

## RAG — Retrieval-Augmented Generation

When `RAG_ENABLED=true`, source documents are chunked and only the top-k most relevant passages are sent to the LLM as context (instead of the full document).

Five retrieval strategies are available:

| Retriever              | Description              | Extra packages                                |
| ---------------------- | ------------------------ | --------------------------------------------- |
| `tfidf`                | TF-IDF cosine similarity | None (default)                                |
| `bm25`                 | Okapi BM25               | None                                          |
| `sentence_transformer` | Dense embeddings         | `pip install sentence-transformers`           |
| `faiss`                | FAISS ANN + embeddings   | `pip install faiss-cpu sentence-transformers` |
| `chroma`               | ChromaDB vector store    | `pip install chromadb`                        |

Per-test retrieval metrics are logged and stored: chunks retrieved, scores, hit@k, MRR, context coverage.

---

## Comparison runs

Compare multiple retriever strategies side-by-side automatically:

```bash
python scripts/compare_runs.py
```

Retriever combinations are defined in `config/comparison.yaml`. The script produces:

- A per-run JSON + HTML report
- A `reports/comparison/comparison_*.html` table showing average metric scores for each strategy with the best-performing configuration highlighted

---

## Dashboard

Aggregate all previous runs into a single dashboard:

```bash
python scripts/generate_dashboard.py
# → reports/dashboard.html
```

Shows: total runs, total tests, average pass rate, average metric scores, execution history, and provider comparison across all runs.

---

## Project structure

```
.env                          Single source of truth for all configuration
.env.example                  Template — copy to .env and fill in values
config/
  comparison.yaml             Retriever strategies to compare (runs only)
docs/
  architecture.md             Technical design and decisions
  learning-journal.md         Day-by-day build log
sample_data/
  documents/                  HR policy source documents (ground truth)
  prompts/                    YAML test cases (question + expected answer)
scripts/
  run_tests.py                Run a single test suite
  compare_runs.py             Run retriever comparison
  generate_dashboard.py       Build the aggregated dashboard
src/ai_orchestrator/
  config/                     Settings (from .env) + config dataclasses
  evaluators/                 Pluggable evaluators (hallucination, relevance,
                              faithfulness, llm_judge)
  loaders/                    Document + prompt loader + document chunker
  models/                     Dataclasses for every domain object
  providers/                  Pluggable LLM providers (Gemini, Hugging Face)
  reporting/                  JSON / HTML / PDF / Comparison / Dashboard reports
  retrievers/                 BaseRetriever + TF-IDF / BM25 / ST / FAISS / Chroma
  runners/                    TestRunner + ComparisonRunner
  services/                   ExecutionService (full pipeline)
tests/                        193 unit + integration tests
```

---

## Tech stack

- Python 3.10+, dataclasses, `pytest` (193 tests)
- Google Gemini API / Hugging Face Inference API
- PyYAML for test-case definitions
- Playwright for PDF report rendering
- **Optional:** `sentence-transformers`, `faiss-cpu`, `chromadb` for dense retrievers

---

## Sample data — fully data-driven

The framework is entirely data-driven. You can change, add, or replace anything in `sample_data/` **without touching any code or config**.

| Change                          | What to do                                                  |
| ------------------------------- | ----------------------------------------------------------- |
| Replace source documents        | Drop new `.md` / `.txt` files into `sample_data/documents/` |
| Edit or add test questions      | Edit a YAML file in `sample_data/prompts/`                  |
| Use a different test suite      | Set `TEST_SUITES=path/to/suite.yaml` in `.env`              |
| Run multiple suites in one pass | `TEST_SUITES=suite_a.yaml,suite_b.yaml` (comma-separated)   |

**Minimum YAML structure for a test suite:**

```yaml
tests:
  - id: my_test
    question: "What is the policy on X?"
    source_document: sample_data/documents/policy.md
    expected_answer: "Optional — used for RAG retrieval quality metrics"
    use_rag: false # optional, defaults to false
```

Swap the entire domain (HR → Legal → Medical → Finance) by replacing the documents and YAML — no code, no evaluator, no config file change required.

---

## CI

Tests run automatically on every push and pull request via GitHub Actions (`.github/workflows/tests.yml`).

- Python 3.12, pip caching
- All 193 tests run on push — **no real API key required** (all tests mock the LLM provider)
- `GEMINI_API_KEY` is set to a placeholder in CI; it is never actually called

---

## Status

Active learning project — not production-ready. See `docs/architecture.md` for a full honest assessment of what's implemented vs. planned.
