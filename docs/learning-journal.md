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

## Learning Entry: Hand-rolling the evaluators before reaching for DeepEval/Ragas

Date: 30 July 2026

This is my first AI engineering project. Before wiring in DeepEval or Ragas for hallucination/relevance/faithfulness scoring, I deliberately built naive versions of each metric myself — stopword filtering + set intersection between answer and context.

Reasoning: I wanted to understand what these metrics are actually measuring under the hood before trusting a library's black-box score. Now that I've felt where the naive approach breaks (it penalizes correct paraphrasing, and can miss hallucinations that reuse source vocabulary), I have a much better basis for evaluating whether DeepEval/Ragas are actually doing something meaningfully different, rather than just calling an API I don't understand.

Decision: keep the `BaseEvaluator` interface as-is; swap the internals for DeepEval/Ragas next, so this becomes a refactor rather than a rewrite.

## Learning Entry: Getting a full engineering review of the codebase

Date: 30 July 2026

Had the codebase reviewed end-to-end (architecture, code quality, testing, security, AI-readiness, DevOps). Key things I learned:

- A duplicate `ProviderRegistry`/`ProviderFactory` had crept into the codebase (one in `providers/__init__.py`, one in `providers/registry.py` + `factory.py`) — different parts of the code used different copies. Easy to miss, exactly the kind of thing a second set of eyes catches.
- Found a real bug: my Gemini provider was catching rate-limit (429) errors and returning them as if they were a valid model answer, which would then get scored by the evaluators as a real (failing) response — silently corrupting results during quota-limited runs.
- My documentation had drifted out of sync with my code in both directions — some docs described finished work as "planned," and the README implied DeepEval/Ragas were already integrated when they weren't yet.
- Biggest takeaway: the _scaffolding_ (pluggable providers/evaluators, config-driven execution, structured results) was solid, but the actual evaluation logic behind it needs to grow up before I call this framework production-ready. That's exactly the next milestone.

Consolidated seven docs down to two (`architecture.md`, `learning-journal.md`) so there's one technical source of truth and one narrative log, instead of six overlapping planning documents.
