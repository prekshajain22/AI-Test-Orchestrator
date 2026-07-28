# Engineering Decision Log

## Decision 001

### Decision

Python will be the primary programming language.

### Why?

- Large AI ecosystem
- Excellent testing libraries
- Strong community support

---

## Decision 002

### Decision

The framework will support multiple LLM providers.

### Why?

Avoid vendor lock-in.

---

## Decision 003

### Decision

Evaluation logic will be independent from model providers.

### Why?

Makes the framework extensible.

## Decision 004

### Decision

Application configuration will be managed through environment variables.

### Why?

- Keeps secrets out of source control.
- Simplifies switching between environments.
- Aligns with common production deployment practices.
