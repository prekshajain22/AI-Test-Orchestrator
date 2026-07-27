# AI Evaluation Matrix

| Test Type         | Question                            | Metric        |
| ----------------- | ----------------------------------- | ------------- |
| Fact Checking     | Is annual leave 25 days?            | Faithfulness  |
| Policy Compliance | Is approval required?               | Accuracy      |
| Hallucination     | Does AI invent remote days?         | Hallucination |
| Completeness      | Are all eligibility rules included? | Completeness  |
| Privacy           | Is medical information protected?   | Safety        |

---

# Evaluation Result Example

Question:

How many remote days can employees work?

AI Response:

Employees can work remotely 3 days per week.

Evaluation:

Faithfulness:
FAILED

Reason:

The source document does not specify a number of remote working days.

Final Result:

FAILED - Hallucination detected
