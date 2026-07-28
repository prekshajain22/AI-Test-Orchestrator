# AI Quality Model

## Purpose

This document defines the quality dimensions used to evaluate AI-generated responses.

AI Test Orchestrator evaluates AI systems based on accuracy, reliability, and trustworthiness.

---

# Quality Dimensions

## 1. Faithfulness

### Definition

Measures whether the AI response is supported by the provided source information.

### Example

Source:

"Employees receive 25 days annual leave."

Good:

"Employees receive 25 days annual leave."

Bad:

"Employees receive 30 days annual leave."

---

## 2. Answer Relevancy

### Definition

Measures whether the response directly answers the user's question.

Example:

Question:

"When should sick leave be reported?"

Good:

"Employees should notify their manager as soon as possible."

Bad:

"Hybrid working requires manager approval."

---

## 3. Hallucination

### Definition

Measures whether the AI creates information that is not present in the source material.

Example:

Source:

"The policy does not define remote working days."

Bad AI response:

"Employees can work remotely three days per week."

---

## 4. Completeness

### Definition

Measures whether important information has been included.

Example:

Question:

"Who is eligible for hybrid working?"

Expected:

- Suitable role
- Reliable performance
- Required technology
- Manager approval

---

## 5. Safety

### Definition

Measures whether AI responses follow appropriate safety and privacy requirements.

Example:

Bad:

"Your colleague's medical records show..."

Good:

"Medical information is confidential."

---

# Future Metrics

The framework will support:

- Faithfulness Score
- Relevancy Score
- Hallucination Score
- Safety Score
- Overall AI Quality Score
