
# 开发协作流程

## 1. 文档目的

本文档记录本项目当前稳定的协作方式，帮助未来的开发者或 AI agent 在新会话中快速延续同一套工作节奏。

本项目不是一次性把所有代码写完，而是采用：

```text
SDD 明确需求
→ TDD 写失败测试
→ 最小实现
→ 聚焦验证
→ Git 分支和 PR
→ 合并后进入下一条用户故事
```

## 2. 核心原则

1. 先写清楚行为，再写代码。
2. 每个 feature branch 只做一条清晰的 TDD slice。
3. 先让测试红，再做最小实现让测试绿。
4. 每次代码改动后，立即运行最窄相关测试。
5. PR 要小、可解释、可验证。
6. 合并前不要把多个不相关 feature 混在一个分支里。
7. 不提前加入未经验证的运行时依赖。
8. 文档、测试、代码要一起演进。

## 3. 新会话开始时的检查步骤

每次重新打开会话或换 agent，先执行以下检查：

```bash
git status --short --branch
git log --oneline --decorate -5
```

然后阅读：

1. `spec/project-architecture.md`
2. `spec/user-stories.md`
3. `spec/test-plan.md`
4. `spec/development-workflow.md`
5. `.github/copilot-instructions.md`

如果当前分支不是从最新 `main` 开出来的，要先修正分支基线，避免把旧 main 的内容和新 feature 混在一起。

## 4. 每条 Feature 的标准流程

### 4.0 `start next feature` 触发规则

当用户说：

```text
start next feature
next feature
开始下一个 feature
新建 feature branch
```

AI agent 应先执行 feature-start git workflow，不要直接改代码。

标准动作：

```bash
git status --short --branch
git switch main
git pull --ff-only
git switch -c feat/<feature-name>
make test
```

如果工作区不干净，先停止并汇报当前改动，不要切分支，也不要自动 stash 或丢弃用户改动。

如果用户没有给出 feature 名称，先询问一个短分支名，例如：

```text
feat/chunk-storage-jsonl
feat/vector-indexing
feat/hybrid-retrieval
```

### 4.1 确认前一个 PR 已合并

在 GitHub 上确认上一条 feature PR 已经 merge 到 `main`。

本地同步：

```bash
git switch main
git pull --ff-only
```

### 4.2 新建 feature branch

分支名使用清晰的 feature 名称：

```bash
git switch -c feat/<feature-name>
```

示例：

```text
feat/pdf-ingestion
feat/pdf-file-loader
feat/upload-flow
feat/hybrid-retrieval
feat/multi-query-retrieval
feat/citation-generation
feat/evaluation-metrics
```

### 4.3 确认绿色基线

开始写新测试前，先确认现有测试是绿的：

```bash
make test
```

如果只做 ingestion 相关功能，也可以先跑：

```bash
make test-ingestion
```

### 4.4 写失败测试

先写测试，不先写实现。

测试应来自：

- `spec/user-stories.md` 的验收标准
- `spec/test-plan.md` 的测试清单

测试命名要表达行为，而不是实现细节。

### 4.5 运行测试确认红灯

运行最窄测试，确认失败原因符合预期：

```bash
make test-ingestion
```

或：

```bash
make test
```

如果失败原因不是预期行为，先修正测试或重新理解需求，不要急着实现。

### 4.6 最小实现

只写让当前测试通过所需的最小代码。

避免在同一条 TDD slice 中顺手做：

- 大重构
- 未计划的抽象
- UI 美化
- 新依赖扩张
- 未来功能预实现

### 4.7 运行聚焦测试和全量测试

实现后先跑最窄测试：

```bash
make test-ingestion
```

通过后跑全量测试：

```bash
make test
```

### 4.8 检查 diff

提交前检查当前变更：

```bash
git status --short --branch
git diff
```

确认没有误提交：

- `.env`
- API key
- private PDF
- generated index
- downloaded model
- `__pycache__`
- `.pytest_cache`

## 5. Commit 规范

推荐 commit 类型：

```text
chore: 项目初始化、工具、脚手架
ci: CI workflow
configs: 配置变更，若需要可使用
contracts: 数据契约或接口约定，若需要可使用
docs: 文档变更
test: 测试变更
feat: 功能实现
fix: bug 修复
refactor: 不改变行为的重构
```

当前项目常用格式：

```bash
git commit -m "test: add PDF file loader tests"
git commit -m "feat: add PDF file loader"
git commit -m "ci: add Python 3.12 test workflow"
git commit -m "docs: document PyPDF2 migration note"
```

如果一条 branch 内既有 CI 又有功能，优先拆成两条 commit：

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add Python 3.12 test workflow"

git add <feature files>
git commit -m "feat: add <feature>"
```

## 6. Push 和 PR 流程

推送当前分支：

```bash
git push -u origin <branch-name>
```

PR title 与主要 commit 保持一致。

PR description 推荐结构：

```md
## Summary

简要说明这条 PR 完成了什么。

## Changes

- 列出主要代码、测试、文档、依赖变化

## Validation

Ran:

```bash
make test
```

Result:

```text
xx passed
```

## Notes

记录重要取舍、warning、后续迁移计划或非本 PR 范围。
```

## 7. CI 策略

项目使用 GitHub Actions 在 PR 和 push 到 `main` 时运行测试。

CI 约束：

- 使用 Python 3.12
- 安装 `requirements.txt`
- 安装 `requirements-dev.txt`
- 运行 `make PYTHON=python test`

如果本地 `make test` 不绿，不要期待 CI 通过。

## 8. 依赖管理流程

不要提前加入大而全的依赖。

新增运行时依赖时，必须满足：

1. 对应实现已经进入当前 TDD slice。
2. 已在 Python 3.12 环境验证。
3. 已写入 `requirements.txt`。
4. 如果依赖有 warning 或迁移风险，要写入 `spec/project-architecture.md`。

当前例子：

- `PyPDF2==3.0.1` 是 PDF file loader 进入实现时才加入。
- PyPDF2 deprecation warning 已记录，未来可迁移到 `pypdf`。

## 9. 当前已形成的 TDD Slice 顺序

已经完成：

1. `DocumentChunk` ingestion model
2. PDF reader ingestion
3. PDF file loader

建议后续顺序：

1. uploaded PDF → build request
2. chunk splitting strategy
3. ingestion summary for multiple PDFs
4. citation source object generation
5. BM25 retrieval
6. FAISS vector store integration
7. hybrid retrieval score fusion
8. multi-query deduplication
9. optional rerank flow
10. evaluation metrics

## 10. 分支基线注意事项

如果新 branch 开错了，比如从旧 `main` 开出，可能会看到 spec/src/tests 变成 untracked 文件。

处理方式：

1. 先用 `git status --short --branch` 和 `git branch -vv` 确认。
2. 如有用户未提交改动，先 stash 或手动保护。
3. 同步 `main`：

```bash
git switch main
git pull --ff-only
```

4. 删除错误 branch 并从最新 main 重建。
5. 必要时只恢复真正有内容差异的用户修改，不恢复 `__pycache__` 等缓存文件。

## 11. 与 AI Agent 协作时的偏好

用户偏好：

- 先讨论清楚是否需要 branch、commit、PR。
- 每条 TDD slice 保持小步推进。
- 重要工程决策要写入 spec，避免新会话遗忘。
- 代码实现前要有测试或明确验收标准。
- 如果发现分支/基线异常，先修 Git 状态，不要继续写代码。
- 回答可以中英混合，但项目文档和 UI 默认中文。
