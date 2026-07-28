# System Architecture

## Overview

AI Test Orchestrator is a modular AI Quality Engineering framework designed to test and evaluate LLM-based applications.

The framework separates test cases, document sources, LLM providers, evaluation engines, and reporting components to support scalable AI testing.

The architecture is designed to support multiple LLM providers and evaluation strategies.

---

# Current Implementation Architecture

The current implemented pipeline:
Test Cases
(YAML)

    |
    v

PromptTestCase Model

    |
    v

Document Loader

    |
    v

Source Document Context

    |
    v

LLM Provider
(Gemini)

    |
    v

AI Response

---

# Implemented Components

## Prompt Test Cases

Responsible for:

- Defining AI test scenarios
- Storing questions
- Maintaining expected answers
- Linking source documents

Current format:

- YAML-based test cases

---

## Document Loader

Responsible for:

- Loading source documents
- Providing ground truth context for AI responses

Current supported format:

- Markdown documents

---

## LLM Provider Layer

The provider layer abstracts interaction with AI models.

Current provider:

- Gemini

Design supports future providers:

- Hugging Face
- OpenAI
- Azure OpenAI

---

# Planned Architecture Components

## Evaluation Engine

Future component responsible for automated AI quality measurement.

Planned metrics:

- Hallucination detection
- Faithfulness
- Answer relevancy
- Context precision
- Bias detection
- Toxicity detection

Potential frameworks:

- DeepEval
- Ragas

---

## Reporting Layer

Future component responsible for generating execution reports.

Planned outputs:

- JSON results
- HTML reports
- PDF reports

---

## CI/CD Integration

Future capability:

- Automated AI regression testing
- Scheduled test execution
- Pipeline quality gates

---

# Future Enhancements

- Multiple LLM comparison
- AI regression testing
- Cost analysis
- Performance metrics
- Dashboard visualization
