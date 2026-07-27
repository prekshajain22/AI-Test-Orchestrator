# System Architecture

## Overview

AI Test Orchestrator is designed as a modular AI Quality Engineering framework.

The framework separates prompt execution, AI providers, evaluation metrics, reporting, and test cases into independent components.

This allows additional LLM providers or evaluation frameworks to be added with minimal changes.

---

## High-Level Architecture

```

Test Cases
(YAML / JSON)

↓

Prompt Runner

↓

LLM Provider
(OpenAI / Hugging Face / Azure OpenAI)

↓

Evaluation Engine
(DeepEval / Ragas)

↓

Reports
(JSON / HTML / PDF)

↓

CI/CD

```

---

## Components

### Prompt Runner

Responsible for:

- Reading prompt test cases
- Executing prompts
- Collecting responses

---

### LLM Provider

Supports multiple providers.

Examples:

- Hugging Face
- OpenAI
- Azure OpenAI

---

### Evaluation Engine

Measures:

- Hallucination
- Answer Relevancy
- Faithfulness
- Context Precision
- Bias
- Toxicity

---

### Reporting

Produces

- HTML Report
- JSON Report
- PDF Report

---

## Future Enhancements

- Multiple LLM Comparison
- AI Regression Testing
- Cost Analysis
- Performance Metrics
- Dashboard
