# AI Quality Engineering Learning Journal

## Day 1

Created project repository.

---

## Day 2

Defined project vision.

Created README.

---

## Day 3

Designed initial architecture.

Created engineering documentation.

Lessons learned:

Good software starts with planning before implementation.

## Day 4

Created AI test scenarios using YAML.

Learned that AI testing requires structured test data containing:

- User question
- Expected behaviour
- Source document
- Validation criteria

This is similar to traditional automation test data design but requires semantic evaluation rather than only exact comparison.

## Day 5

Created enterprise-style HR policy documents to act as ground truth data.

Learned that AI evaluation requires trusted reference material.

Added hallucination scenarios where the expected behaviour is that the AI should refuse to invent information not present in the source document.

## Day 6

Defined AI evaluation criteria.

Learned that AI testing requires measuring multiple quality dimensions:

- Faithfulness
- Relevancy
- Hallucination
- Completeness
- Safety

Unlike traditional automation, AI testing focuses on quality scoring rather than only pass/fail validation.

## Learning Entry: Building First LLM Test Pipeline

Date: 28 July 2026

Learnings:

- LLM testing is similar to traditional QA validation.
- Expected vs Actual comparison still applies.
- Source documents provide the ground truth.
- Provider abstraction allows multiple AI models.
- Hallucination testing requires checking unsupported claims.

Technical Skills Practiced:

- Python package structure
- API integration
- Environment variables
- YAML test data
- LLM prompting
