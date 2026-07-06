# V2 功能计划：LangChain + LangGraph 增强版

## 1. 文档目的

本文档记录 PolicyPilot-RAG V2 的功能扩展计划。

V2 在现有原生 RAG 实现基础上，新增三个页面，使用 LangChain 和 LangGraph 实现高级 RAG 技术，
与原生版本形成横向对比，同时展示框架使用能力和底层机制理解。

---

## 2. 总体架构决策

### 2.1 页面方案

采用 Streamlit 多页面方案，新页面与现有页面共用同一个知识库：

```
app/
  streamlit_app.py             ← 原生实现版（现有，不改动）
  pages/
    1_langchain_rag.py          ← LangChain 高级 RAG 页（含 LangGraph）
    2_knowledge_health.py       ← 知识库健康检查页
    3_knowledge_extraction.py   ← 对话知识沉淀页
```

### 2.2 知识库共享方案

原有 PDF 上传、分块、embedding 流程不重做。LangChain 页面从现有 `chunks.jsonl` 构建
LangChain 格式的 FAISS 索引，存至 `runtime/vector_index_lc/`。

```
原有流程（不变）
  PDF → pdfplumber → chunk_splitter → chunks.jsonl → runtime/vector_index/

LangChain 初始化（一次性，点击触发）
  chunks.jsonl → FAISS.from_texts() → runtime/vector_index_lc/
```

### 2.3 Rerank 决策

V2 不实现本地 Rerank 模型（BGE reranker 约 500MB，不适合 Streamlit Cloud 免费版）。
后续可通过 DashScope `gte-rerank` API 增加，不影响当前设计。

---

## 3. 页面一：LangChain 高级 RAG（含 LangGraph）

### 3.1 页面目标

展示使用 LangChain 组件 + LangGraph 编排实现的高级 RAG 能力，与原生版形成对比。

### 3.2 功能模块

#### A. 检索模式（侧边栏切换）

| 模式 | 实现 | 说明 |
|------|------|------|
| 标准向量检索 | `VectorStoreRetriever` | 基线 |
| 多路查询 MultiQuery | `MultiQueryRetriever` | LLM 生成 3 个查询变体，合并去重 |
| 混合检索 Hybrid | `EnsembleRetriever` | BM25 + Vector，alpha=0.5 |
| 自适应检索 Adaptive | LangGraph `StateGraph` | 见 3.3 节 |

#### B. 查询改写（侧边栏开关）

使用 LCEL chain：

```python
rewrite_chain = prompt | llm | StrOutputParser()
```

开启后：在问题输入框下方显示改写后的查询，然后用改写结果做检索。

#### C. 多轮对话 Memory

使用 `ConversationSummaryBufferMemory`：

```python
memory = ConversationSummaryBufferMemory(
    llm=llm,
    max_token_limit=800,   # 约 4-5 轮中文对话
    memory_key="chat_history",
    return_messages=True,
    output_key="answer",
)
```

**Memory 机制**：
- 最近对话原文保留在 buffer（不超过 800 tokens）
- 超出后旧轮次被 LLM 压缩成摘要
- 再次超出时将新积累的轮次与旧摘要合并生成新摘要
- 最终注入 LLM 的 context = `[摘要] + [最近原文轮次]`

#### D. UI 设计（对话模式）

由于引入 Memory，UI 从单次问答升级为对话模式：

```
侧边栏
  ├── 知识库状态
  ├── 检索模式选择（Standard / MultiQuery / Hybrid / Adaptive）
  ├── 查询改写开关
  └── Memory 状态（当前轮次 / 是否有摘要）

主区域
  ├── 对话历史（st.chat_message 气泡）
  ├── 最新一轮的检索详情（expander：改写后的查询 / 检索路径 / 证据）
  └── st.chat_input 输入框（固定底部）
```

会话历史存储在 `st.session_state.messages`，每次渲染时重放所有气泡。

### 3.3 LangGraph：Adaptive RAG 自适应检索路由

**设计目标**：先用 BM25 检索，若证据质量不足则升级为混合检索，仍不足则返回 fallback。

**状态定义**：

```python
class AdaptiveRagState(TypedDict):
    question: str
    rewritten_query: str
    retrieved_chunks: list
    evidence_score: float
    retrieval_mode: str   # "bm25" | "hybrid" | "fallback"
    answer: str
    chat_history: list
```

**图结构**：

```
[rewrite_query]
      │
[retrieve_bm25]
      │
[evaluate_quality]
      ├── score >= 0.3 ──▶ [generate_answer]
      └── score < 0.3
            │
      [retrieve_hybrid]
            │
      [evaluate_quality]
            ├── score >= 0.3 ──▶ [generate_answer]
            └── score < 0.3  ──▶ [return_fallback]
```

**UI 呈现**：在检索详情 expander 中显示实际走的路径，例如：

```
检索路径：BM25 → 证据不足 → 升级混合检索 → 命中
```

---

## 4. 页面二：知识库健康检查

### 4.1 页面目标

使用 LLM 检查当前知识库对典型问题的覆盖情况，找出知识空白。

### 4.2 实现方案

LCEL chain + `JsonOutputParser`：

```python
health_check_chain = prompt | llm | JsonOutputParser()
```

**流程**：
1. 用户输入或系统生成若干测试问题
2. 对每个问题执行检索，将检索结果 + 问题一起发给 LLM
3. LLM 返回结构化 JSON：覆盖率评分、缺失知识点、建议

**输出 schema**：

```json
{
  "coverage_score": 0.75,
  "missing_knowledge": [
    {
      "query": "客户经理季度考核权重是多少？",
      "missing_aspect": "季度考核权重细则",
      "importance": "高",
      "suggested_content": "建议补充季度考核章节"
    }
  ],
  "completeness_analysis": "知识库覆盖了投诉处理和月度考核，缺少季度考核细则"
}
```

---

## 5. 页面三：对话知识沉淀

### 4.1 页面目标

从历史问答对话中提取有价值的知识点，支持扩充知识库。

### 4.2 实现方案

LCEL chain + 结构化输出：

```python
extraction_chain = prompt | llm | JsonOutputParser()
```

**流程**：
1. 展示本次会话（或历史会话）的问答记录
2. 点击「提取知识」，LLM 分析对话并返回结构化知识点
3. 用户可选择将提取结果追加至 `chunks.jsonl`（触发重建索引）

**输出 schema**：

```json
{
  "extracted_knowledge": [
    {
      "knowledge_type": "事实",
      "content": "客户经理被投诉一次且核查属实，当月绩效扣5分",
      "confidence": 0.9,
      "keywords": ["投诉", "绩效", "扣分"],
      "category": "考核规则"
    }
  ],
  "conversation_summary": "用户询问了投诉处理对绩效的影响",
  "user_intent": "了解投诉处罚细则"
}
```

---

## 6. 新增依赖

```
# 新增至 requirements.txt
langchain-core
langchain-community
langchain-text-splitters
langgraph
```

约新增 ~50MB 安装体积，Streamlit Cloud 免费版可接受。

---

## 7. 开发顺序

按依赖关系从底层到顶层推进：

```
Step 1  共享知识库加载层
        src/app_services/langchain_knowledge_base.py
        - 从 chunks.jsonl 构建 LangChain FAISS 索引
        - @st.cache_resource 缓存

Step 2  LangChain RAG 页基础版
        pages/1_langchain_rag.py
        - 标准向量检索 + Memory + 对话 UI

Step 3  高级检索模式
        - MultiQueryRetriever
        - EnsembleRetriever (Hybrid)
        - Query Rewrite LCEL chain

Step 4  LangGraph Adaptive RAG
        src/app_services/adaptive_rag_graph.py
        - StateGraph 定义
        - 节点函数
        - 条件边

Step 5  知识库健康检查页
        pages/2_knowledge_health.py

Step 6  对话知识沉淀页
        pages/3_knowledge_extraction.py
```

每个 Step 对应一个独立的 feature branch，遵循现有 TDD 工作流。

---

## 8. 测试策略

| 模块 | 测试重点 |
|------|---------|
| `langchain_knowledge_base.py` | 从 chunks.jsonl 构建索引，返回正确数量的文档 |
| `adaptive_rag_graph.py` | 各节点函数的输入输出结构；条件边路由逻辑（mock retriever） |
| Health check chain | JSON schema 验证；空知识库的 fallback |
| Extraction chain | JSON schema 验证；空对话的处理 |

LLM 输出不测试具体文字，只测试结构和路由逻辑（同现有测试原则）。
