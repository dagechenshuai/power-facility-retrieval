# 电力设施智能检索平台

面向电力技术文档、图片和三维场景查询，提供文档导入（8002）和智能查询（8003）两个 FastAPI 服务。

查询通过模型选择检索工具，融合向量、BM25、HyDE、业务接口或联网资料。重排后，模型根据证据选择回答、补查、追问或停止；最多执行两轮检索。代码使用 LangGraph Checkpoint 保存节点状态，用 SQLite 事件日志支持 SSE 重放。

## 先运行一个本地示例

需要 Python 3.12 和 uv，在仓库根目录执行：

```powershell
uv sync --locked
uv run python -m evaluation.demo
uv run pytest -q
```

示例使用 `examples/demo-documents.json` 中的合成资料，真实执行 BM25 入库和父块检索，不需要 API Key、GPU 或数据库服务。输出示例见 `examples/demo-output.json`。它只验证本地检索，不包含真实模型决策，也不能用于证明企业数据上的效果指标。

## 完整服务运行

```powershell
Copy-Item .env.example .env
```

填写配置并准备依赖。完整启动步骤、各服务职责和请求示例见 [运行说明](docs/standards/running.md)。分别在两个终端启动：

```powershell
uv run uvicorn web.api.import_service:app --host 127.0.0.1 --port 8002
uv run uvicorn web.api.query_service:app --host 127.0.0.1 --port 8003
```

浏览器访问 `http://127.0.0.1:8002` 导入资料，访问 `http://127.0.0.1:8003` 查询。当前内置页面适合本地无鉴权演示；配置 `AGENT_API_KEY` 后请通过支持鉴权请求头的 API 客户端调用。

## 状态与恢复

`POST /query` 设置 `is_stream=true` 后取得 `task_id`，通过 `GET /stream/{task_id}` 接收 `ready`、`delta`、`final`、`error` 事件。`ready.reset=true` 表示新一轮答案开始，客户端需清空旧的部分答案；`final.answer` 是最终完整文本。

- `Last-Event-ID` 控制断线重放，每个订阅者独立读取持久日志。
- `GET /tasks/{task_id}` 返回任务状态、结果和决策记录。
- `POST /tasks/{task_id}/resume` 恢复失败或中断任务。
- 非流式执行失败返回 HTTP 502；空答案不会记为成功。
- `status` 表示执行状态；`result_status` 表示证据判断，例如 `answered`、`needs_clarification`、`insufficient_evidence`、`incomplete`。

当前建议每个服务只运行一个 worker。节点检查点不等于外部副作用恰好执行一次，恢复时可能重新调用模型或业务接口；多 worker 抢占、任务租约和完整进程崩溃实验仍待验证。`/ready` 当前只检查 SQLite，不代表其他依赖已就绪。

## 验证材料

- [评测和证据边界](docs/evaluation.md)：已有自动化验证、历史指标需要的证据及未验证事项。
- [故障修复说明](docs/fix/query-reliability.md)：失败状态、SSE、并行状态合并及上下文预算。
- [查询决策设计](docs/standards/query-decisions.md)：工具参数、决策循环、上限与当前限制。
- `examples/tests.xml`：本次实际生成的 pytest 报告。

运行真实 RAG 评测前，需要自行准备脱敏知识库及人工标注的 `evaluation/golden.jsonl`：

```powershell
uv run python -m evaluation.live_rag_eval evaluation/golden.jsonl --k 10 --output evaluation/result.jsonl
uv run python -m evaluation.rag_evaluator evaluation/result.jsonl --k 10
```

`evaluation/dataset.example.jsonl` 只是格式说明。公开仓库尚未提供简历历史指标的原始逐题记录，因此这里不将测试通过率解释为真实检索准确率。
