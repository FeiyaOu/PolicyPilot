# PolicyPilot RAG

基于 RAG（检索增强生成）的银行内部制度问答助手示例项目。从 PDF 文档摄入到带引用的答案生成，展示完整的 RAG 系统生命周期。采用 TDD 工作流与清晰的服务层架构开发。

## 在线演示

已部署至 Streamlit Community Cloud —— 上传银行制度 PDF，即可进行带来源引用的政策问答。

---

## 功能概览（v1）

| 层级 | 能力 |
|------|------|
| 文档摄入 | PDF 文本提取（pdfplumber）、按页切分、中文感知分块、chunk 元数据（来源文件 + 页码） |
| 存储 | JSONL chunk 存储、FAISS 向量索引、BM25 稀疏索引 |
| 检索 | 纯向量检索、纯 BM25 检索、混合检索（加权分数融合） |
| 生成 | 基于证据的答案生成、引用溯源、证据不足时的降级回答 |
| 界面 | 知识库构建页、制度问答页、侧边栏展示 Embedding 与 LLM 配置状态 |

---

## 技术栈

| 组件 | 选型 |
|------|------|
| UI 框架 | Streamlit |
| PDF 提取 | pdfplumber |
| 向量索引 | FAISS（`faiss-cpu`） |
| 稀疏检索 | `rank-bm25` + jieba 分词 |
| Embedding 模型 | DashScope `text-embedding-v4`（1024 维） |
| LLM | DashScope `deepseek-v3` |
| 测试框架 | pytest |
| Python 版本 | 3.12 |

---

## 项目结构

```
app/
  streamlit_app.py              # UI 入口
src/
  app_services/                 # 服务层，连接 UI 与核心逻辑
    build_service.py
    embedding_provider_config.py
    llm_provider_config.py
    knowledge_base_loader.py
    retrieval_service.py
    ui_answer_service.py
    vector_index_build_page_service.py
  ingestion/                    # PDF → chunks
    pdf_ingestion.py
    chunk_splitter.py
    chunk_store.py
  retrieval/                    # FAISS + BM25 + 混合检索
    vector_index.py
    bm25.py
    hybrid.py
  generation/                   # 提示词、证据、引用、答案
    prompt_builder.py
    evidence.py
    citations.py
    answer_generator.py
    answer_contract.py
runtime/                        # 运行时生成，已加入 .gitignore
  processed/chunks.jsonl
  vector_index/
spec/                           # 架构设计、用户故事、测试计划
tests/                          # 与 src/ 结构对应的测试目录
```

---

## 本地运行

### 1. 安装依赖

```bash
make install-dev
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```env
DASHSCOPE_API_KEY=sk-your-key-here
POLICYPILOT_LLM_PROVIDER=dashscope
POLICYPILOT_LLM_MODEL=deepseek-v3
POLICYPILOT_EMBEDDING_PROVIDER=dashscope
POLICYPILOT_EMBEDDING_MODEL=text-embedding-v4
POLICYPILOT_EMBEDDING_DIMENSION=1024
```

### 3. 启动应用

```bash
make run-ui
```

### 4. 运行测试

```bash
make test
```

---

## 部署到 Streamlit Community Cloud

1. 将代码推送到 GitHub
2. 访问 [share.streamlit.io](https://share.streamlit.io)，创建新应用：
   - **Main file path**：`app/streamlit_app.py`
   - **Branch**：`main`
3. 在 **Advanced settings → Secrets** 中填入：

```toml
DASHSCOPE_API_KEY = "sk-your-key-here"
POLICYPILOT_LLM_PROVIDER = "dashscope"
POLICYPILOT_LLM_MODEL = "deepseek-v3"
POLICYPILOT_EMBEDDING_PROVIDER = "dashscope"
POLICYPILOT_EMBEDDING_MODEL = "text-embedding-v4"
POLICYPILOT_EMBEDDING_DIMENSION = "1024"
```

4. 点击 Deploy。每次重新部署后，需重新上传 PDF 并点击**构建知识库**（`runtime/` 目录不提交至代码仓库）。

---

## RAG 流水线

```
上传 PDF
  └─▶ pdfplumber 文本提取
        └─▶ 按页切分（800 字/chunk，150 字重叠，中文分隔符）
              └─▶ chunks.jsonl（持久化）
                    ├─▶ FAISS 向量索引（text-embedding-v4，每批最多 10 条）
                    └─▶ BM25 索引（jieba 分词）

用户提问
  └─▶ embed_query → FAISS top-k        （向量分数）
  └─▶ jieba 分词 → BM25 top-k          （BM25 分数）
  └─▶ 混合分数融合                      （alpha × 向量 + (1-alpha) × BM25）
        └─▶ 证据评估（最低分数阈值过滤）
              └─▶ deepseek-v3 生成带引用的答案
```

---

## 架构原则

- **服务层隔离**：所有核心逻辑位于 `src/`，Streamlit 仅负责 UI 编排
- **确定性行为 TDD**：chunk 元数据、分数融合、证据选择、Provider 配置均有测试覆盖
- **协议式 Mock**：PDF 读取器、Embedding 客户端、LLM 客户端均通过依赖注入——测试中不发起真实 API 调用
- **显式降级处理**：API Key 缺失、证据为空、Provider 报错均返回结构化的 fallback，不向外抛出原始异常
