# PolicyPilot RAG - Project Architecture

## 1. Purpose

PolicyPilot RAG is an interview-ready RAG showcase project. It turns the isolated RAG scripts in the existing notes into one coherent product: an enterprise policy knowledge assistant for bank staff.

The goal is not only to answer questions from documents. The goal is to demonstrate a complete RAG system lifecycle:

- document ingestion
- chunking and source tracing
- dense vector retrieval
- BM25 sparse retrieval
- hybrid retrieval
- query rewriting
- multi-query retrieval
- reranking
- cited answer generation
- knowledge-base health checks
- conversation knowledge extraction
- knowledge-base version comparison
- retrieval evaluation and regression testing

## 2. Development Method

We will use SDD first, then TDD.

In this project, SDD means Specification-Driven / Scenario-Driven Development:

1. Define the product scenario.
2. Define user roles and workflows.
3. Define the RAG pipeline behavior.
4. Define acceptance criteria before implementation.
5. Convert important behavior into tests.
6. Implement with TDD where practical.

TDD will be used mainly for deterministic parts:

- document chunk metadata handling
- query rewrite result parsing
- hybrid score fusion
- deduplication of multi-query results
- rerank result ordering
- knowledge health report schema
- version diff logic
- evaluation metric calculation

LLM outputs are probabilistic, so tests around LLM behavior should validate structure, fallbacks, and pipeline behavior rather than exact wording.

## 3. Interview Positioning

The project should be presented as:

> An enterprise policy RAG assistant that helps bank employees query internal policy documents with citations, compare retrieval strategies, monitor knowledge-base health, and evaluate knowledge-base versions.

The key interview message:

> I started from a basic PDF RAG assistant, then improved retrieval quality with query rewriting, multi-query search, BM25 + vector hybrid retrieval, and reranking. After that I added RAG operations: citation tracing, health checks, conversation-based knowledge extraction, version comparison, and regression evaluation.

## 4. Topic And Scenario

### Topic

Enterprise Policy Knowledge Copilot for bank branch staff.

### Primary Scenario

A bank customer manager needs to ask questions about internal assessment policies, complaint handling, evaluation rules, and branch operating procedures.

Example user questions:

- 客户经理被投诉一次会扣多少分？
- 投诉会不会影响年度评聘？
- 客户经理每年评聘申报时间是什么时候？
- 如果客户经理既有业绩问题又有投诉记录，考核怎么处理？
- 上面说的投诉处理流程有没有例外情况？

The system should answer with source evidence, not just generate text.

## 5. Product Shape

The first showcase version will use Streamlit as the demo and observability interface.

Streamlit is used because it is fast for AI demos, dashboards, and interview presentation. It is not the core architecture. The core RAG logic should live in modular Python services under `src/`, so the UI can later be replaced by FastAPI + React/Vue if needed.

Suggested app sections:

1. Ask Policy Assistant
2. Retrieval Lab
3. Knowledge Health
4. KB Version Compare
5. Evaluation Report

## 6. High-Level Architecture

```text
Streamlit UI
    |
    v
Application Service Layer
    |
    +-- Query Rewrite Service
    +-- Retrieval Service
    +-- Rerank Service
    +-- Answer Generation Service
    +-- Knowledge Health Service
    +-- Conversation Knowledge Service
    +-- Evaluation Service
    |
    v
Storage Layer
    |
    +-- Raw documents
    +-- Processed chunks
    +-- FAISS index for demo retrieval
    +-- BM25 index for sparse retrieval
    +-- Metadata and evaluation artifacts
```

## 7. Core RAG Pipeline

### Step 1: Ingestion

Input sources:

- PDF policy documents
- optional markdown/text policy files
- optional conversation logs

Processing:

- extract text
- split into chunks
- keep metadata:
  - document name
  - page number
  - chunk id
  - section title if available
  - version id

Chunk splitting strategy:

- split page by page instead of concatenating the full PDF first
- preserve `source_file` and `page_number` on every chunk
- use Chinese-aware separators before falling back to character-level splitting
- default `chunk_size` is 800 characters
- default `chunk_overlap` is 150 characters
- store chunk-local metadata such as `chunk_index`, `start_char`, and `end_char`

Default separators:

```python
["\n\n", "\n", "。", "；", "，", ".", ";", ",", " ", ""]
```

Reason:

The previous learning examples used a LangChain-style recursive splitter with `chunk_size=1000`, `chunk_overlap=200`, and mostly English separators. This project uses the same general idea but adapts it for Chinese bank policy PDFs and stable source citation. Page-aware splitting avoids fragile page-number reconstruction after full-document splitting.

Processed chunk storage:

- the build service can persist final chunks to `runtime/processed/chunks.jsonl`
- each JSONL record uses the `DocumentChunk.to_dict()` shape
- the JSONL file is the local source of truth for later vector indexing and BM25 indexing
- `runtime/` is ignored by Git because it contains generated or private artifacts

### Step 2: Query Understanding

The query rewrite layer should handle:

- context-dependent questions
- vague references such as “这个”, “它”, “上面说的”
- comparative questions
- multi-intent questions
- rhetorical or emotional questions
- web-search-needed detection for time-sensitive questions

For the first local version, web search can be represented as a decision and rewritten search strategy. Actual web search integration can be a later extension.

### Step 3: Retrieval

The project should support multiple retrieval modes:

- vector-only retrieval with FAISS
- BM25-only retrieval
- hybrid retrieval with weighted score fusion
- multi-query retrieval
- hybrid + multi-query retrieval
- hybrid + multi-query + rerank

The Retrieval Lab should make these modes visible and comparable.

Hybrid score fusion starts as a deterministic scoring contract before concrete retrievers are added:

```text
fused_score = alpha * vector_score + (1 - alpha) * bm25_score
```

- `alpha=1.0` means vector-only ranking
- `alpha=0.0` means BM25-only ranking
- missing scores default to `0.0`
- results are sorted by `fused_score` descending
- upstream FAISS distances and BM25 raw scores should be normalized before fusion

This follows the same idea as the learning examples in `RAG 高级技术/CASE-高效召回`, where BM25 scores are normalized, FAISS distances are converted to similarity scores, and both are merged with a configurable vector weight.

Multi-query retrieval also starts with a deterministic contract before adding LLM query generation:

- normalized query variants always include the original user query
- empty query variants fall back to the original query
- duplicate retrieved hits are merged by `chunk_id`, not by text content
- when the same chunk appears from multiple query variants, keep the highest score
- returned results keep the original query and query variants for observability

BM25 sparse retrieval is part of the basic retrieval layer, not an advanced rerank feature:

- load processed chunk records from `runtime/processed/chunks.jsonl`
- tokenize Chinese policy text with `jieba`
- build an Okapi BM25 index with `rank_bm25.BM25Okapi`
- return chunk IDs, normalized BM25 scores, content, and source metadata
- keep reranking out of the default retrieval path until the basic vector/BM25/hybrid flow is usable

Vector retrieval uses FAISS as the local demo vector index:

- build FAISS indexes from processed chunk records in `runtime/processed/chunks.jsonl`
- use `faiss.IndexFlatL2` for the first exact-search implementation
- persist vector indexes under ignored `runtime/indexes/` paths
- store chunk metadata beside the FAISS index so search results can cite source files and pages
- convert FAISS L2 distances into normalized similarity scores before hybrid fusion
- tests use deterministic embeddings to validate index behavior without network or model dependencies
- production/demo embeddings can later use Chinese embedding models such as `bge-m3`, `gte-qwen2`, or DashScope embeddings

The retrieval service is the application-layer bridge used by UI and answer-generation flows:

- supports vector-only retrieval
- supports BM25-only retrieval
- supports hybrid retrieval through `fuse_hybrid_scores`
- returns one unified result shape with vector score, BM25 score, fused score, content, source file, page number, and metadata
- keeps reranking out of the default basic retrieval path

### Step 4: Reranking

Use a BGE reranker such as `BAAI/bge-reranker-base` for candidate reranking.

Pipeline:

```text
query
→ generate candidates with hybrid/multi-query retrieval
→ rerank candidates with cross-encoder
→ keep top final chunks
→ generate cited answer
```

### Step 5: Answer Generation

The answer generator should:

- answer only from retrieved context
- cite source document and page/chunk
- say when evidence is insufficient
- expose retrieved evidence in the UI

Citation generation is deterministic and does not parse LLM output text:

- build citation objects from retrieval result metadata
- include source file, page number, and contributing chunk IDs
- deduplicate citations by source file and page number
- use fallback fields when source metadata is missing

Evidence sufficiency is checked before answer generation:

- return an evidence-insufficient result when retrieval returns no chunks
- return an evidence-insufficient result when the best retrieval score is below a configured threshold
- do not fabricate citations when evidence is insufficient
- preserve the original question and a retrieval summary for UI/debug display
- pass selected evidence chunks and deterministic citations forward when evidence is sufficient

Answer output uses a stable deterministic contract before LLM integration:

- include the original question
- expose an answer status for answer-ready vs evidence-insufficient states
- include selected context snippets with source metadata and scores
- include deterministic citations and retrieval summary
- return a fixed fallback message when evidence is insufficient
- allow generated answer text to be attached later without changing the UI contract

Basic answer generation is provider-driven:

- skip provider calls when evidence is insufficient
- pass the original question and selected context snippets to an injected answer provider
- attach generated answer text to the stable answer output contract
- preserve citations, contexts, and retrieval summary from evidence review
- keep concrete LLM/API clients outside the core contract for deterministic tests

## 8. Knowledge Operations

### Knowledge Health

The system should analyze a knowledge base for:

- missing knowledge
- outdated knowledge
- conflicting knowledge
- overall health score
- suggested improvements

This is important because it shows the project is more than a chatbot.

### Conversation Knowledge Extraction

The system should extract durable knowledge from conversations:

- facts
- procedures
- policy clarifications
- common Q&A
- precautions

It should avoid storing temporary user-specific needs as permanent knowledge.

### Version Management

The system should support comparing knowledge-base versions:

- added chunks
- removed chunks
- modified chunks
- unchanged chunks
- retrieval accuracy before/after
- regression test pass rate

## 9. Storage Decision

### Initial Demo Storage

Use FAISS for the first version.

Reason:

- simple local setup
- fast retrieval
- good for small demo datasets
- easy to explain retrieval mechanics
- suitable for GitHub + Streamlit prototype

Do not commit large generated indexes or private vector databases to Git.

Recommended Git strategy:

```text
Commit:
- sample public/synthetic documents
- ingestion scripts
- evaluation test cases
- README and architecture docs

Do not commit:
- .env
- API keys
- private PDFs
- large vector indexes
- downloaded reranker models
```

Recommended `.gitignore` entries later:

```gitignore
.env
__pycache__/
*.pyc
runtime/
indexes/
vector_db/
vector_db_hybrid/
models/
```

### Runtime Artifact Storage Policy

Uploaded files and generated indexes are runtime artifacts. They are stored under `runtime/` locally, ignored by Git, and can be regenerated from safe sample PDFs. In production, replace local runtime storage with object storage, a database, and a vector database.

Recommended local layout:

```text
data/
  raw/                         # safe sample/default PDFs; may be committed only if non-private

runtime/
  uploads/                     # user-uploaded PDFs during local/demo runtime
  processed/
    chunks.jsonl               # chunk text and metadata; source of truth for local rebuilds
  indexes/
    faiss/                     # local FAISS vector index files
    bm25/                      # BM25 cache/index artifacts
  evaluation/                  # generated evaluation reports
```

Storage rules:

- `data/raw/` is for safe sample documents that can be used to rebuild a demo index.
- `runtime/uploads/` is for files uploaded while the app is running.
- `runtime/processed/chunks.jsonl` should be the local source of truth for chunk text and citation metadata.
- FAISS and BM25 indexes are generated artifacts and should be rebuildable from `chunks.jsonl`.
- `runtime/` must not be committed to Git.
- Do not upload private or real internal bank documents to public Streamlit deployments.

Streamlit deployment note:

Streamlit Community Cloud provides a filesystem, but it should be treated as ephemeral demo storage rather than durable backend storage. For the first public demo, use safe sample PDFs and regenerate indexes when needed. For production, store PDFs in object storage, metadata/evaluation records in a database, and vectors in pgvector, Qdrant, Milvus, or another production vector store.

### Production Upgrade Path

The retrieval layer should be designed so FAISS can later be replaced.

Potential production options:

- PostgreSQL + pgvector: best when relational metadata, permissions, conversations, evaluations, and vector search should live together.
- Qdrant: strong dedicated vector database with metadata filtering and simple deployment.
- Milvus/Zilliz: stronger for large-scale vector infrastructure.
- Chroma: useful local developer-friendly vector store, but less impressive as an enterprise production choice.

Interview explanation:

> I used FAISS for a lightweight reproducible demo. For production, I would use PostgreSQL with pgvector if I need relational metadata and operational records in one system, or Qdrant/Milvus if vector search scale and filtering are the main concern.

## 10. Dependency Management Decision

The project currently keeps runtime dependencies intentionally minimal.

Do not add a formal `requirements.txt` too early. Streamlit, LangChain, FAISS, DashScope, reranker models, and other runtime dependencies should be added only when the corresponding RAG pipeline implementation begins and the dependency versions have been verified in the Python 3.12 environment.

Current dependency approach:

- `pyproject.toml` stores project metadata and pytest configuration.
- `requirements-dev.txt` stores development/test dependencies such as `pytest`.
- `Makefile` provides stable local commands for development and testing.
- `requirements.txt` will be added later for verified runtime dependencies.

Reason:

- avoid committing unverified dependencies
- keep the early SDD/TDD phase clean
- prevent FAISS/Python-version conflicts
- make future deployment dependencies easier to explain and reproduce

When implementation reaches Streamlit, LangChain, FAISS, DashScope, or rerank integration, add the verified runtime dependencies to `requirements.txt` and document the tested Python version.

### PDF Parsing Dependency Note

The first PDF file-loader implementation uses `PyPDF2==3.0.1` because the existing learning examples in this workspace already use PyPDF2 and the dependency has been verified in Python 3.12 for the current TDD slice.

PyPDF2 emits a deprecation warning recommending migration to `pypdf`. This warning does not block the current TDD flow, but it should be tracked as a future cleanup item.

Future migration plan:

- keep `PyPDF2==3.0.1` for the current implementation slice
- consider replacing it with `pypdf` before the ingestion module becomes stable
- add/adjust tests before migration to ensure `ingest_pdf_file()` behavior remains unchanged
- update `requirements.txt` only after the replacement is verified in Python 3.12

## 11. Deployment Decision

The first deployable demo should use:

- GitHub repository for code
- Streamlit UI for product demo
- Streamlit Community Cloud or Hugging Face Spaces for public access if reachable by the target audience

Important note:

A GitHub repo alone does not create a clickable web app. The Streamlit app needs to be deployed to a hosting service to produce a public URL.

For China-based interviewers, Streamlit Community Cloud accessibility may be inconsistent depending on network conditions. A safer plan is:

1. Keep the app runnable locally with one command.
2. Provide screenshots/GIFs in README.
3. Optionally deploy a second mirror using a China-accessible platform if needed.
4. Keep the backend modular so deployment target can change.

## 12. Proposed Project Structure

```text
PolicyPilot-RAG/
  README.md
  project-architecture.md
  requirements.txt
  .env.example
  .gitignore

  data/
    raw/
    processed/
    sample_queries/

  indexes/
    faiss/
    bm25/

  src/
    ingestion/
      pdf_loader.py
      chunker.py
      metadata.py

    query_rewrite/
      rewriter.py
      schemas.py

    retrieval/
      base.py
      faiss_store.py
      bm25_store.py
      hybrid.py
      multi_query.py

    reranking/
      bge_reranker.py

    generation/
      answerer.py
      prompts.py

    kb_ops/
      health_check.py
      conversation_extractor.py
      versioning.py

    evaluation/
      test_cases.py
      metrics.py
      regression.py

    app_services/
      rag_pipeline.py
      retrieval_lab.py
      health_dashboard.py

  app/
    streamlit_app.py

  scripts/
    ingest.py
    ask.py
    evaluate.py
    health_check.py
```

## 13. First Acceptance Criteria

Before implementation, the first version should satisfy these behaviors:

1. A user can upload or use sample policy documents.
2. The system can build a searchable index.
3. The user can ask a policy question and receive an answer with citations.
4. The user can inspect retrieved chunks and scores.
5. The user can compare retrieval modes.
6. The system can run a small evaluation set.
7. The system can produce a basic knowledge health report.
8. The project can run locally from documented commands.

## 14. Review Questions

The following decisions are accepted for version 1:

1. Project name: `PolicyPilot-RAG`.
2. Main scenario: bank internal policy assistant.
3. UI: Streamlit for the first demo.
4. Storage: FAISS for version 1, with pgvector/Qdrant as the production upgrade path.
5. Language: Chinese UI and Chinese README.
6. Dataset: start with the existing bank policy PDF; more PDFs can be added later.
7. Deployment priority: local demo first, public web demo second.

## 15. User Interaction Plan

The user should feel they are using a policy assistant, but the interviewer should be able to inspect the RAG pipeline behind every answer.

### 14.1 First-Time Setup Flow

1. User opens the Streamlit app.
2. App checks whether an index already exists.
3. If no processed chunks exist, the app asks the user to build one from uploaded PDFs or the configured PDF folder.
4. User clicks `构建知识库`.
5. App extracts text, chunks documents, stores metadata in `runtime/processed/chunks.jsonl`, and reports indexing results.
6. BM25 retrieval is rebuilt from `chunks.jsonl` when the answer service reloads.
7. FAISS/vector index build remains a later UI slice.

Indexing result should show:

- processed document count
- chunk count
- embedding model
- vector store path
- BM25 index status
- warnings for pages with missing text

### 14.2 Main Question-Answer Flow

1. User enters a policy question in Chinese.
2. User optionally provides conversation context.
3. App rewrites the query if needed.
4. App generates multiple query variants.
5. App retrieves candidates through the selected retrieval strategy.
6. App reranks candidates if rerank mode is enabled.
7. App generates an answer from the final context.
8. App displays citations and the evidence chunks.

The answer page should show:

- final answer
- source PDF name and page number
- confidence or evidence sufficiency indicator
- rewritten query
- generated query variants
- retrieved chunks
- retrieval scores
- rerank scores when available

The first minimal Streamlit UI is a local demo surface:

- keep Streamlit as a thin rendering layer under `app/streamlit_app.py`
- route question answering through app services in `src/`
- build `runtime/processed/chunks.jsonl` from uploaded PDFs and `data/raw` PDFs
- show answer-ready and evidence-insufficient states
- show citations, selected evidence chunks, and retrieval summary
- load `runtime/processed/chunks.jsonl` when available
- build BM25 retrieval from loaded chunk records for the first real local UI path
- load an existing FAISS vector index when `runtime/vector_index` and an embedding provider are available
- configure DashScope embeddings from environment variables for vector query embedding
- show only whether the embedding provider is configured; never display API keys
- configure DashScope LLM generation from environment variables for grounded answer generation
- fall back to the local demo answer provider when the LLM API key is missing
- expose available retrieval modes in the UI, starting with BM25 and enabling vector/hybrid when FAISS is loaded
- show a clear missing/empty knowledge-base state when chunks are not available
- do not include rerank in the first UI slice

Embedding provider configuration uses:

- `DASHSCOPE_API_KEY`
- `POLICYPILOT_EMBEDDING_PROVIDER`, default `dashscope`
- `POLICYPILOT_EMBEDDING_MODEL`, default `text-embedding-v4`
- `POLICYPILOT_EMBEDDING_DIMENSION`, default `1024`

LLM provider configuration uses:

- `DASHSCOPE_API_KEY`
- `POLICYPILOT_LLM_PROVIDER`, default `dashscope`
- `POLICYPILOT_LLM_MODEL`, default `qwen-plus`

LLM requests are built from a grounded system/user prompt. The prompt includes the user question and numbered evidence snippets with source file and page number. The assistant must answer in Chinese using only supplied evidence and state uncertainty when evidence is insufficient.

### 14.3 Retrieval Lab Flow

The Retrieval Lab is for demonstrating technical depth.

User selects one question and compares:

- vector-only retrieval
- BM25-only retrieval
- hybrid retrieval
- hybrid + multi-query retrieval
- hybrid + multi-query + rerank

For each mode, the app should show:

- top-k chunks
- scores
- source pages
- overlap between methods
- which method produced the final answer context

### 14.4 Knowledge Health Flow

The user runs a health check over the current knowledge base.

The app should show:

- missing knowledge
- outdated knowledge
- conflicting knowledge
- health score
- suggested fixes

This section demonstrates that the project handles RAG operations, not only RAG answering.

### 14.5 Evaluation Flow

The user runs a small test set of policy questions.

The app should show:

- test question
- expected evidence or expected answer keywords
- retrieved chunks
- pass/fail result
- recall@k
- answer citation presence
- regression result after a knowledge-base update

## 16. LangChain Decision

Use LangChain selectively, not as the whole architecture.

LangChain is useful in this project for:

- PDF/document loader integrations when helpful
- text splitting with `RecursiveCharacterTextSplitter`
- FAISS vector store integration
- document object conventions
- optional model wrappers

Do not hide the core RAG logic inside a single LangChain chain.

The following logic should stay explicit in our own modules so it is easy to test and explain in interviews:

- query rewrite routing
- multi-query generation and deduplication
- BM25 retrieval
- hybrid score fusion
- rerank orchestration
- citation metadata handling
- health-check schemas
- version comparison
- evaluation metrics

Interview explanation:

> I use LangChain for reliable building blocks such as text splitting and FAISS integration, but I keep the retrieval pipeline explicit because the purpose of this project is to demonstrate how RAG works: query rewriting, hybrid retrieval, reranking, citations, evaluation, and knowledge operations.

## 17. Future V2 Agentic RAG Plan

LangGraph will not be used in version 1.

Version 1 should keep the RAG pipeline explicit, simple, and testable. The first goal is to demonstrate retrieval mechanics and RAG quality improvement, not to build an autonomous agent framework.

LangGraph may be introduced in version 2 if the project evolves into an agentic RAG system with stateful branching, retries, evidence checks, and human review.

Potential version 2 workflow:

```text
用户提问
→ 判断问题类型
→ 判断是否需要 Query 改写
→ 判断是否需要多意图拆分
→ 执行检索
→ 判断证据是否足够
→ 如果证据不足，扩大检索或切换检索策略
→ 如果仍然不足，标记为缺失知识
→ Rerank 候选文档
→ 生成答案
→ 检查引用是否完整
→ 如果引用不足，重新生成或提示证据不足
→ 提取可沉淀的对话知识候选
→ 等待人工审核是否写入知识库
```

LangGraph would be useful for:

- evidence sufficiency checks
- conditional retrieval retries
- fallback from vector retrieval to hybrid retrieval
- multi-intent query routing
- citation validation and answer regeneration
- human-in-the-loop knowledge updates
- conversation knowledge deposition workflow
- knowledge health repair workflow

Interview explanation:

> I considered LangGraph, but did not use it in version 1 because the first milestone should make the RAG pipeline transparent and easy to test. LangGraph is a good version 2 choice when the assistant needs stateful branching, evidence sufficiency checks, retry loops, and human approval for knowledge-base updates.
