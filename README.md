# AI Test Orchestrator

A learning project building an AI Quality Engineering framework from first principles: testing LLM-based applications the way a QA engineer would test any other system — with structured test cases, ground-truth source documents, automated evaluation, and repeatable reports.

This is my first hands-on AI engineering project, built and documented as I go. See [`docs/learning-journal.md`](docs/learning-journal.md) for the day-by-day build log, and [`docs/architecture.md`](docs/architecture.md) for the technical design.

## What this actually does today

Given a set of questions, a source document (ground truth), and an LLM provider:

1. Loads test cases from YAML (`sample_data/prompts/`)
2. Loads the matching source document as context
3. Sends the question + context to an LLM provider (Gemini or Hugging Face)
4. Scores the response against three evaluators (see below)
5. Generates JSON, HTML, and PDF reports with pass/fail rates and per-metric breakdowns

## Evaluators — hand-rolled by design, for now

The three evaluators (`hallucination`, `relevance`, `faithfulness`) are **hand-rolled lexical-overlap heuristics** — stopword filtering + set intersection between the answer and the source context. No embeddings, no LLM-as-judge, no DeepEval/Ragas yet.

That's intentional at this stage: I wanted to understand _what these metrics actually measure_ by implementing naive versions myself before reaching for a library that does it for me. **DeepEval and Ragas integration is planned next** (see roadmap in `docs/architecture.md`) — this README will be updated when that lands, and the current heuristics will move behind the same evaluator interface rather than being thrown away.

## Setup

```bash
git clone https://github.com/prekshajain22/AI-Test-Orchestrator.git
cd AI-Test-Orchestrator
pip install -r requirements.txt
cp .env.example .env   # then fill in GEMINI_API_KEY (or HF_API_KEY if using Hugging Face)
python scripts/run_tests.py
```

![Tests](https://github.com/prekshajain22/AI-Test-Orchestrator/actions/workflows/tests.yml/badge.svg)

Reports are written to `reports/json/`, `reports/html/`, and `reports/pdf/`.

## Configuration

All runs are controlled by `config/execution.yaml`:

```yaml
provider:
  name: gemini
test_suites:
  - sample_data/prompts/hr_questions.yaml
evaluators:
  - hallucination
  - relevance
  - faithfulness
reports:
  - json
  - html
  - pdf
```

## Project structure

```
config/          Central run configuration
docs/            architecture.md, learning-journal.md
sample_data/     Source documents + prompt test cases used as ground truth
src/ai_orchestrator/
  config/        Settings + config loading
  evaluators/    Pluggable evaluation strategies (hallucination, relevance, faithfulness)
  loaders/       Document + prompt-test-case loaders
  models/        Dataclasses for every domain object
  providers/     Pluggable LLM providers (Gemini, Hugging Face)
  reporting/     JSON/HTML/PDF report generation
  runners/       Top-level orchestration
  services/      End-to-end execution service
tests/           ~30 unit tests across loaders, evaluators, providers, reporting
```

## Tech stack

- Python 3.10+, dataclasses, `pytest`
- Google Gemini API / Hugging Face Inference API
- PyYAML for test-case and config definitions
- Playwright for PDF report rendering
- **Planned:** DeepEval, Ragas, GitHub Actions CI

## Status

Early-stage learning project — not production-ready. See `docs/architecture.md` for a current honest assessment of what's implemented vs. planned, including known limitations.
