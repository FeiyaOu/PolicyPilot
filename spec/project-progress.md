# PolicyPilot-RAG Project Progress

Last updated: 2026-07-05

## 1. Purpose Of This File

This file is a handoff note for future agents and developers.

Read it at the start of a new session together with:

1. `spec/project-architecture.md`
2. `spec/user-stories.md`
3. `spec/test-plan.md`
4. `spec/development-workflow.md`
5. `.github/copilot-instructions.md`

The goal is to preserve project context: what has already been built, what is still missing, and what feature should probably come next.

## 2. Current Project Goal

`PolicyPilot-RAG` is an interview-ready RAG showcase project for a Chinese bank internal policy assistant.

The project should demonstrate a complete RAG lifecycle, not only a simple PDF chatbot:

- PDF ingestion
- page-aware chunk splitting
- chunk metadata and source tracing
- processed chunk storage
- vector retrieval with FAISS
- BM25 sparse retrieval
- hybrid retrieval
- query rewriting
- multi-query retrieval
- reranking
- cited answer generation
- evidence-insufficient fallback
- retrieval evaluation
- knowledge-base health checks
- knowledge-base version comparison

## 3. Development Workflow

The project uses SDD before TDD.

Feature workflow:

```text
git status --short --branch
→ git switch main
→ git pull --ff-only
→ git switch -c feat/<feature-name>
→ make test
→ write a focused failing test
→ implement the smallest code needed
→ run focused tests
→ run full tests
→ inspect diff
→ commit / push / PR
```

When the user says `start next feature`, follow the feature-start workflow documented in `spec/development-workflow.md`.

If the worktree is dirty, stop and summarize the changes before switching branches. Do not auto-stash or discard user work.

## 4. Environment And Commands

Python:

```text
Python 3.12
Preferred command: /usr/local/bin/python3
```

Common commands:

```bash
make test
make test-ingestion
make test-retrieval
```

Current expected full test status after `feat/hybrid-score-fusion`:

```text
31 passed, 1 warning
```

The warning is the known PyPDF2 deprecation warning.

## 5. Completed Work

### 5.1 Project Setup And Specs

Completed:

- project architecture spec
- user stories
- TDD test plan
- development workflow document
- project-level Copilot instructions
- GitHub Actions CI
- Makefile test commands
- `.gitignore` for runtime/generated/private artifacts

Important files:

```text
spec/project-architecture.md
spec/user-stories.md
spec/test-plan.md
spec/development-workflow.md
.github/copilot-instructions.md
.github/workflows/ci.yml
Makefile
```

### 5.2 Ingestion Foundation

Completed:

- `DocumentChunk` model
- stable `chunk_id` generation
- chunk metadata support
- PDF reader ingestion
- PDF file ingestion
- warnings for pages with no extractable text
- ingestion summary

Important files:

```text
src/ingestion/models.py
src/ingestion/pdf_ingestion.py
tests/ingestion/test_document_chunk.py
tests/ingestion/test_pdf_ingestion.py
tests/ingestion/test_pdf_file_loader.py
```

### 5.3 Upload And Build Flow

Completed:

- uploaded PDF abstraction
- raw/default PDF document input abstraction
- build request abstraction
- build service that ingests raw and uploaded documents through the same ingestion boundary

Important files:

```text
src/app_services/upload_flow.py
src/app_services/build_service.py
tests/app_services/test_upload_flow.py
tests/app_services/test_build_service.py
```

### 5.4 Page-Aware Chunk Splitting

Completed:

- deterministic page-aware chunk splitter
- Chinese-aware separators
- default `chunk_size=800`
- default `chunk_overlap=150`
- chunk metadata: `chunk_index`, `start_char`, `end_char`
- PDF ingestion now uses the splitter

Important files:

```text
src/ingestion/chunk_splitter.py
tests/ingestion/test_chunk_splitter.py
src/ingestion/pdf_ingestion.py
```

Design note:

The implementation was inspired by the LangChain-style recursive splitter in `RAG 高级技术/CASE-高效召回`, but it stays local and deterministic for now so page metadata and citations remain stable.

### 5.5 Processed Chunk JSONL Storage

Completed:

- write processed chunks to JSONL
- read JSONL records back in file order
- preserve UTF-8 Chinese text
- build service can optionally persist chunks to `runtime/processed/chunks.jsonl`

Important files:

```text
src/ingestion/chunk_store.py
tests/ingestion/test_chunk_store.py
src/app_services/build_service.py
```

Design note:

`runtime/processed/chunks.jsonl` is the local source of truth for later vector indexing and BM25 indexing. `runtime/` is ignored by Git.

### 5.6 Hybrid Score Fusion

Completed or currently in PR depending on merge state:

- deterministic hybrid score fusion contract
- weighted formula:

```text
fused_score = alpha * vector_score + (1 - alpha) * bm25_score
```

- `alpha=1.0` means vector-only ranking
- `alpha=0.0` means BM25-only ranking
- missing scores default to `0.0`
- results sort by `fused_score` descending
- retrieval-focused Makefile target

Important files:

```text
src/retrieval/hybrid.py
tests/retrieval/test_hybrid.py
Makefile
```

Design note:

This was inspired by `RAG 高级技术/CASE-高效召回/3-chatpdf-faiss-HybridSearch.py`, where BM25 scores are normalized, FAISS distances are converted into similarity scores, and both are merged with a configurable vector weight. This project first defines the pure scoring contract before adding concrete retrievers.

## 6. Current Branch / PR Status

At the time this file was created, the local branch was:

```text
feat/hybrid-score-fusion
```

Latest pushed commit:

```text
d8ac40b feat: add hybrid score fusion
```

PR link:

```text
https://github.com/FeiyaOu/PolicyPilot/pull/new/feat/hybrid-score-fusion
```

Before starting the next feature, confirm this PR has been merged to `main`, then run the standard `start next feature` workflow.

## 7. What Still Needs To Be Done

### 7.1 Retrieval Foundation

Recommended next features:

1. `feat/multi-query-dedup`
2. `feat/bm25-retrieval`
3. `feat/vector-indexing`
4. `feat/hybrid-retrieval-service`

Suggested order:

```text
multi-query dedup
→ BM25 retrieval over chunks.jsonl
→ FAISS vector indexing over chunks.jsonl
→ hybrid retrieval service using fuse_hybrid_scores
```

Reason:

- multi-query dedup is pure deterministic logic and easy to TDD
- BM25 can be implemented locally and tested without external APIs
- FAISS introduces dependency and embedding choices, so do it after the retrieval contracts are stable
- hybrid retrieval service should compose existing vector/BM25 results rather than inventing scoring again

### 7.2 Multi-Query Retrieval

Completed or currently in PR depending on merge state:

- query variant normalization
- original query is always included
- empty variants fall back to the original query
- retrieval hits are deduplicated by `chunk_id`
- duplicate chunk strategy is highest score wins
- output keeps original query and query variants for observability

Important files:

```text
src/retrieval/multi_query.py
tests/retrieval/test_multi_query.py
```

Remaining later work:

Needs:

- query variant list schema
- LLM or rule-based query variant generation
- retrieval orchestration that runs each query variant against a retriever
- integration with BM25, vector, or hybrid retrieval

Planned tests:

```text
tests/retrieval/test_multi_query.py
```

### 7.3 BM25 Retrieval

Completed or currently in PR depending on merge state:

- Chinese tokenization with `jieba`
- Okapi BM25 indexing with `rank_bm25.BM25Okapi`
- BM25 retriever can be built from chunk records or `chunks.jsonl`
- search returns chunk ID, normalized BM25 score, content, source file, page number, and metadata
- empty corpus and no-match queries return an empty list

Important files:

```text
src/retrieval/bm25.py
tests/retrieval/test_bm25.py
requirements.txt
```

Remaining later work:

Needs:

- retrieval service orchestration
- integration with multi-query retrieval
- integration with hybrid score fusion
- Retrieval Lab display of BM25-only mode

Possible dependency:

```text
rank_bm25
```

Add only when implemented and verified in Python 3.12.

### 7.4 Vector Indexing With FAISS

Needs:

- embedding model decision
- build FAISS index from `runtime/processed/chunks.jsonl`
- persist FAISS index under ignored runtime/index path
- load FAISS index for retrieval
- map FAISS results back to chunk metadata
- convert FAISS distance into normalized similarity score before fusion

Possible dependencies:

```text
faiss-cpu
sentence-transformers or provider-specific embedding client
```

Do not add these before a focused feature branch verifies compatibility.

### 7.5 Hybrid Retrieval Service

Needs:

- vector retriever result contract
- BM25 retriever result contract
- score normalization contract
- use `fuse_hybrid_scores`
- return top-k chunks with source metadata
- expose retrieval mode and scores for Retrieval Lab

### 7.6 Reranking

Needs:

- rerank result ordering
- preserve chunk metadata
- optional rerank mode, not default mandatory path
- candidate empty-list fallback

Possible model:

```text
BAAI/bge-reranker-base
```

Planned tests:

```text
tests/reranking/test_rerank_order.py
```

### 7.7 Cited Answer Generation

Needs:

- answer input schema from retrieved chunks
- citation object generation from chunk metadata
- duplicate citation removal
- evidence-insufficient fallback
- avoid parsing citations from LLM output text

Planned tests:

```text
tests/generation/test_citations.py
tests/generation/test_insufficient_evidence.py
```

### 7.8 Streamlit Demo UI

Needs:

- knowledge-base build page
- ask policy assistant page
- retrieval lab page
- evaluation report page
- optional system settings page

Rule:

Streamlit should stay as UI orchestration only. Core behavior must remain in `src/` and be testable without Streamlit.

### 7.9 Evaluation

Needs:

- small hand-written evaluation set
- source recall metric
- citation check
- pass rate calculation
- report generation

Planned tests:

```text
tests/evaluation/test_metrics.py
```

### 7.10 Knowledge Operations

Later-stage features:

- knowledge health report schema
- missing/outdated/conflicting knowledge detection
- knowledge-base version comparison
- conversation knowledge extraction

These should come after the first minimal RAG question-answering loop works.

## 8. Recommended Next Feature

If `feat/hybrid-score-fusion` has been merged, the recommended next feature is:

```text
feat/multi-query-dedup
```

Why:

- it is pure deterministic logic
- it fits the current TDD style
- it prepares for multi-query retrieval without requiring LLM calls yet
- it keeps retrieval work moving before adding FAISS/BM25 dependencies

Expected files:

```text
src/retrieval/multi_query.py
tests/retrieval/test_multi_query.py
```

Expected behaviors:

- duplicate chunk IDs are merged
- deduplication is based on `chunk_id`, not content text
- duplicate strategy is explicit, preferably keep the highest score
- empty query variants fall back to the original query
- output includes original question and query variants for observability

## 9. RAG Learning Examples To Reuse For Inspiration

For retrieval-related features, read examples under:

```text
RAG 高级技术/CASE-高效召回/
```

Most relevant files:

```text
1-MultiQueryRetriever使用.py
2-chatpdf-faiss-MultiQueryRetriever.py
3-chatpdf-faiss-HybridSearch.py
4-chatpdf-faiss-HybridSearch-Rerank.py
```

Use them for technique inspiration, especially:

- query expansion / multi-query retrieval
- BM25 + vector hybrid retrieval
- score normalization
- reranking flow

Do not blindly copy the notebooks/scripts into this project. Convert useful ideas into small, tested modules under `src/`.

## 10. Important Project Rules For Future Agents

- Use Python 3.12.
- Keep generated/private artifacts out of Git.
- Keep Streamlit thin; put core logic in `src/`.
- Prefer small TDD slices.
- First red test should usually be a single test file or test function.
- Use module-level tests such as `make test-ingestion` or `make test-retrieval` after the focused test is green.
- Run `make test` before commit.
- Do not add LangChain, FAISS, BM25, reranker, or embedding dependencies until a focused feature needs and verifies them.
- For LLM flows, test schemas, fallbacks, metadata propagation, and orchestration boundaries rather than exact wording.
