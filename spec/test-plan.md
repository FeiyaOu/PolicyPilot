# TDD 测试计划

## 1. 文档目的

本文档把 `user-stories.md` 中“可转化为测试的行为”整理成第一版 TDD 测试计划。

测试计划优先覆盖确定性逻辑，避免一开始测试 LLM 的具体回答文本。LLM 相关能力主要测试输入输出结构、fallback、元数据传递和流程编排。

## 2. 测试原则

1. 先测核心数据结构，再测流程编排。
2. 先测不依赖外部 API 的逻辑，再测需要模型或 embedding 的集成逻辑。
3. Streamlit 页面不直接承载复杂逻辑，测试应主要覆盖 `src/` 中的模块。
4. LLM 输出不可预测，不测试固定措辞，只测试 schema、字段、引用、fallback 和证据链。
5. 每个测试应能说明它保护了哪个用户故事或验收标准。

## 3. 第一阶段测试范围

第一阶段优先覆盖最小可展示闭环：

1. 知识库构建
2. 政策问答助手
3. 检索实验室
4. 基础评估报告

暂不完整测试第二阶段功能：

- 知识库健康检查
- 知识库版本对比
- 对话知识沉淀

但会为这些模块预留测试目录和未来测试项。

## 4. 测试分层

### 4.1 单元测试

用于测试纯逻辑：

- `DocumentChunk` 数据结构
- chunk metadata 生成
- hybrid score 计算
- 多查询去重
- rerank 排序
- 引用来源生成
- 证据不足 fallback
- 评估指标计算

### 4.2 集成测试

用于测试多个模块的组合：

- PDF → chunk → metadata
- uploaded PDF → ingestion request
- mock retriever → answer input context
- retrieval results → citations
- evaluation test cases → evaluation report

### 4.3 端到端冒烟测试

用于确认主要流程可跑通：

- 使用测试 PDF 构建知识库
- 使用 mock 或小型本地索引执行一次问答
- 运行 10 个手写评估问题并生成报告

## 5. 第一批 TDD 测试清单

### 5.1 DocumentChunk 数据结构

目标：后续 FAISS、BM25、引用页码、评估都依赖稳定的 chunk metadata。

建议测试文件：`tests/ingestion/test_document_chunk.py`

测试用例：

1. `DocumentChunk` 应包含 `chunk_id`、`content`、`source_file`、`page_number`、`metadata`。
2. `content` 为空时应被拒绝或标记为无效。
3. `page_number` 应为正整数。
4. `chunk_id` 应可稳定生成，不能依赖随机值。
5. `DocumentChunk` 应可转换为 dict，便于保存和 UI 展示。

验收来源：

- `user-stories.md` 4.5
- `user-stories.md` 5.5

### 5.2 PDF Ingestion 输出格式

目标：保证 PDF 抽取和切分后，每个 chunk 都有可追溯来源。

建议测试文件：`tests/ingestion/test_pdf_ingestion.py`

测试用例：

1. 给定一个测试 PDF，ingestion 应返回非空 chunk 列表。
2. 每个 chunk 都应包含来源文件名。
3. 每个 chunk 都应包含页码。
4. 如果某页无法抽取文本，应返回 warning，不应中断整个流程。
5. ingestion summary 应包含文档数、chunk 数、warning 数。

验收来源：

- `user-stories.md` 4.4
- `user-stories.md` 4.5

### 5.3 PDF 上传进入构建流程

目标：确认 Streamlit 上传文件不会绕过统一 ingestion 逻辑。

建议测试文件：`tests/app_services/test_upload_flow.py`

测试用例：

1. 上传的 PDF 应被转换成统一的 document input 对象。
2. 上传文件和默认 `data/raw` 文件应能合并进入同一次构建请求。
3. 非 PDF 文件应被拒绝并返回用户可理解的错误。
4. 上传文件不应直接写入索引，应先进入 ingestion/build service。

验收来源：

- `user-stories.md` 4.3
- `user-stories.md` 4.4
- `user-stories.md` 13

### 5.4 Hybrid 分数计算

目标：确保混合检索的分数可解释、可复现。

建议测试文件：`tests/retrieval/test_hybrid.py`

测试用例：

1. 当 `alpha=0.5` 时，融合分数应等于向量分数和 BM25 分数的平均。
2. 当 `alpha=1.0` 时，融合分数应只使用向量分数。
3. 当 `alpha=0.0` 时，融合分数应只使用 BM25 分数。
4. 结果应按融合分数降序排列。
5. 缺失某一路分数时应使用默认值，并保持结果可返回。

验收来源：

- `user-stories.md` 6.5
- `user-stories.md` 6.6

### 5.5 多查询去重

目标：多查询召回可能重复命中同一 chunk，必须稳定去重。

建议测试文件：`tests/retrieval/test_multi_query.py`

测试用例：

1. 多个查询返回相同 `chunk_id` 时，只保留一个结果。
2. 去重应基于 `chunk_id`，而不是只基于文本内容。
3. 去重后应保留最高分或最早命中的结果，具体策略需要在实现前固定。
4. 返回结果应包含原始问题和查询变体列表。
5. 空查询变体列表时，应至少使用原始问题执行检索。

验收来源：

- `user-stories.md` 5.5
- `user-stories.md` 6.5
- `user-stories.md` 6.6

## 6. 第二批 TDD 测试清单

### 6.1 Rerank 排序

建议测试文件：`tests/reranking/test_rerank_order.py`

测试用例：

1. Rerank 后结果应按 `rerank_score` 降序排列。
2. Rerank 只作为高级选项，不应在默认 pipeline 中强制执行。
3. Rerank 结果应保留原始 chunk metadata。
4. 当候选文档为空时，应返回空列表，不应报错。

验收来源：

- `user-stories.md` 5.4
- `user-stories.md` 6.6
- `user-stories.md` 13

### 6.2 引用来源生成

建议测试文件：`tests/generation/test_citations.py`

测试用例：

1. 引用列表应从 chunk metadata 生成。
2. 引用应包含 `source_file` 和 `page_number`。
3. 相同来源和页码的引用应去重。
4. 引用生成不应依赖 LLM 输出文本解析。
5. 当 chunk 缺失来源信息时，应返回可理解的 fallback 字段。

验收来源：

- `user-stories.md` 5.4
- `user-stories.md` 5.5
- `user-stories.md` 11

### 6.3 证据不足 fallback

建议测试文件：`tests/generation/test_insufficient_evidence.py`

测试用例：

1. 当检索结果为空时，应返回“证据不足”类结果。
2. 当检索分数低于阈值时，应返回证据不足或低置信提示。
3. 证据不足时不应编造引用。
4. 证据不足结果仍应包含原始问题和检索过程摘要。

验收来源：

- `user-stories.md` 5.4
- `user-stories.md` 5.5
- `user-stories.md` 11

### 6.4 评估指标计算

建议测试文件：`tests/evaluation/test_metrics.py`

测试用例：

1. 当 expected_source 出现在 top-k chunk 中时，source recall 应通过。
2. 当答案包含引用来源时，citation check 应通过。
3. `pass_rate` 应等于通过题数除以总题数。
4. 空测试集应返回明确错误。
5. 第一版测试集应包含 10 个手写问题。

验收来源：

- `user-stories.md` 9.4
- `user-stories.md` 9.5
- `user-stories.md` 13

## 7. 第三批测试清单

这些测试可以在第一版最小闭环完成后补充。

### 7.1 知识库健康检查 schema

建议测试文件：`tests/kb_ops/test_health_report.py`

测试重点：

- 健康报告固定 schema
- 总体健康分范围为 0 到 1
- 缺失、过期、冲突字段必须存在
- LLM JSON 解析失败时返回 fallback report

### 7.2 知识库版本对比

建议测试文件：`tests/kb_ops/test_versioning.py`

测试重点：

- added chunk 检测
- removed chunk 检测
- modified chunk 检测
- unchanged chunk 检测
- 第一版页面显示“开发中”状态

### 7.3 系统设置

建议测试文件：`tests/config/test_settings.py`

测试重点：

- 配置默认值
- API Key 只显示存在或缺失
- 修改 chunk 参数后标记索引需要重建

## 8. 推荐实现顺序

1. `DocumentChunk` 数据结构测试
2. PDF ingestion 输出格式测试
3. 上传 PDF 构建请求测试
4. Hybrid 分数计算测试
5. 多查询去重测试
6. 引用来源生成测试
7. 证据不足 fallback 测试
8. 评估指标计算测试
9. Rerank 排序测试
10. Streamlit 页面冒烟测试

## 9. 第一条 TDD 任务

第一条任务：实现 `DocumentChunk` 数据结构和最小测试。

原因：

- 它是 RAG 系统中最底层的知识单元。
- 后续检索、引用、评估、版本比较都依赖它。
- 该逻辑确定性强，适合作为第一条 TDD 测试。

预期测试文件：

```text
tests/ingestion/test_document_chunk.py
```

预期实现文件：

```text
src/ingestion/models.py
```

第一条测试应验证：

- 能创建合法的 `DocumentChunk`
- 空内容会被拒绝
- 页码必须是正整数
- 能转换为 dict
- `chunk_id` 可稳定生成
