# 电力设施智能检索平台

本项目面向电力设施技术文档、三维模型场景和外部资料检索，包含文档导入（8002）和智能查询（8003）两个 FastAPI 服务。

## 核心能力

- **智能路由**：LangGraph 编排查询流程，由模型自主选择向量、BM25、HyDE、场景和联网检索工具。
- **混合检索**：父子分块结合 BGE-M3、传统 BM25、RRF 与 Cross-Encoder，兼顾语义召回和精确关键词匹配。
- **多模态入库**：MinerU 解析 PDF，视觉模型为图片生成中文描述，MinIO 保存原图并参与检索。
- **流式恢复**：逐 Token SSE、事件持久化、断线补发和 LangGraph Checkpoint 支持任务恢复。

## 启动

```powershell
Copy-Item .env.example .env
uv sync
uv run uvicorn web.api.import_service:app --host 127.0.0.1 --port 8002
uv run uvicorn web.api.query_service:app --host 127.0.0.1 --port 8003
```

生产环境应设置 `AGENT_API_KEY`、明确的 `CORS_ORIGINS`，并把 `data/` 放在持久卷上。`/health` 用于存活探针，`/ready` 用于状态库就绪探针。

## 已实现的 Agent 能力

- LLM `stream()` 产生的每个增量都会写入持久事件表并通过 SSE `delta` 事件即时推送；浏览器根据 SSE `id` 断线补发。
- LangGraph 使用 SQLite Checkpoint，并以 `query:{task_id}`、`import:{task_id}` 隔离线程。重启后进行中的任务标记为 `interrupted`，调用 `POST /tasks/{task_id}/resume` 从检查点继续。
- 文档按父章节生成稳定 `parent_id`，子块参加向量召回，命中后去重回取完整 `parent_content`。
- SQLite FTS5 + jieba 构建独立的传统 BM25 索引，不把 BGE-M3 稀疏向量冒充 BM25。
- 模型通过 `bind_tools(..., tool_choice='auto')` 自主选择向量、BM25、HyDE、场景或联网搜索，可多工具并行。
- 本地图片在入库前逐张调用视觉模型生成中文描述；描述失败会使任务失败并保留恢复点，不会静默跳过。

## 恢复与 SSE

`POST /query` 使用 `is_stream=true` 后返回 `task_id`，客户端连接 `GET /stream/{task_id}`。事件类型为 `ready`、`delta`、`final`、`error`。查询和导入任务均可通过 `GET /tasks/{task_id}` 查看持久状态。

## 自动化测试与评测

```powershell
uv run pytest -q
uv run python -m evaluation.rag_evaluator evaluation/result.jsonl
uv run python -m evaluation.live_rag_eval evaluation/golden.jsonl --output evaluation/result.jsonl
```

单元回归覆盖 SSE 重放、任务状态落盘、Checkpoint 恢复、父子回取、BM25、RRF、Rerank、工具路由和图片描述。端到端评测输出 Recall@K、MRR、nDCG 和答案关键词覆盖率，任一指标低于阈值时进程返回非零退出码，可直接作为 CI 质量门禁。

`evaluation/dataset.example.jsonl` 只是格式示例；上线前必须用实际知识库的父块 ID 建立并评审 `golden.jsonl`。
