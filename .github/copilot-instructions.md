# PolicyPilot RAG Copilot Instructions

## Project Context

This project is `PolicyPilot-RAG`, a Chinese bank internal policy RAG assistant showcase project.

The project should demonstrate a complete RAG lifecycle, not just a simple LLM chat UI:

- PDF ingestion
- chunk metadata and source tracing
- FAISS vector retrieval
- BM25 sparse retrieval
- hybrid retrieval
- query rewriting
- multi-query retrieval
- optional reranking
- cited answer generation
- retrieval evaluation
- future knowledge health and version comparison

## Language Rules

- User-facing UI text must be Chinese.
- README and specification documents should be Chinese unless a section is explicitly meant for interview English wording.
- Code identifiers should use clear English names.
- Comments should be sparse and useful; avoid explaining obvious code.

## Development Method

Use SDD before TDD, then implement incrementally.

Before coding a new feature:

1. Check `spec/project-architecture.md`.
2. Check `spec/user-stories.md`.
3. Check `spec/test-plan.md`.
4. Add or update tests for deterministic behavior.
5. Implement the smallest code needed to pass the focused test.

Prefer TDD for deterministic logic:

- `DocumentChunk` metadata
- PDF ingestion output shape
- hybrid score fusion
- multi-query deduplication
- rerank result ordering
- citation object generation
- evidence-insufficient fallback
- evaluation metric calculation

Do not test exact LLM wording. For LLM-related flows, test schema, fallback behavior, metadata propagation, and orchestration boundaries.

## Python Environment

Use Python 3.12 for this project because FAISS compatibility matters.

Preferred local command:

```bash
/usr/local/bin/python3
```

Use `make test` or `make test-ingestion` where available.

## Dependency Policy

Do not add `requirements.txt` too early.

Current policy:

- `pyproject.toml` stores project metadata and pytest configuration.
- `requirements-dev.txt` stores development/test dependencies.
- `Makefile` stores stable local commands.
- Add `requirements.txt` only when runtime dependencies are actually implemented and verified.

When adding runtime dependencies such as Streamlit, LangChain, FAISS, DashScope, or reranker packages, verify them in Python 3.12 first and document the tested versions.

## Architecture Rules

Keep Streamlit as a UI layer only. Core logic should live under `src/` and be testable without Streamlit.

Use LangChain selectively for infrastructure:

- text splitting
- document conventions
- FAISS integration when useful
- optional model wrappers

Do not hide the core RAG behavior inside a single LangChain chain. Keep these explicit in project modules:

- query rewrite routing
- multi-query generation and deduplication
- BM25 retrieval
- hybrid score fusion
- rerank orchestration
- citation metadata handling
- evaluation metrics

Do not use LangGraph in v1. LangGraph is reserved for a future v2 agentic RAG workflow.

## Storage Rules

Use FAISS for v1 local/demo retrieval.

Do not commit generated indexes, private PDFs, downloaded models, API keys, or `.env` files.

Generated or private artifacts should stay out of Git:

- `indexes/`
- `vector_db/`
- `vector_db_hybrid/`
- `models/`
- `data/processed/`

## Testing Rules

After the first substantive code edit, run the narrowest relevant test.

For ingestion work, run:

```bash
make test-ingestion
```

For all tests, run:

```bash
make test
```

It is acceptable for a new TDD test to fail before the implementation exists. Once implementation begins, keep the feedback loop narrow and focused.

## Git Rules

Use feature branches for meaningful changes.

Recommended commit style:

- `chore: ...` for project setup, specs, tooling
- `test: ...` for tests
- `feat: ...` for implemented features
- `fix: ...` for bug fixes
- `docs: ...` for documentation-only changes

Do not commit generated indexes, model files, `.env`, or private documents.
