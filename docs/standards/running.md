# 完整环境运行说明

## 依赖配置

先执行 `uv sync --locked`，再从 `.env.example` 创建 `.env`。不要提交 `.env`。

| 依赖 | 用途 | 配置 |
| --- | --- | --- |
| 兼容 OpenAI 的模型服务 | 问题解析、工具路由、检索反馈、答案生成 | OPENAI_API_BASE、OPENAI_API_KEY、LLM_DEFAULT_MODEL；需要支持工具调用和结构化输出 |
| Milvus | 向量和设施名称索引 | MILVUS_URL、FACILITY_CHUNKS_COLLECTION、FACILITY_NAME_COLLECTION |
| BGE-M3 | 本地向量编码 | BGE_M3_PATH 或 BGE_M3；默认 CPU，首次可能下载模型 |
| MongoDB | 对话历史 | MONGO_URL、MONGO_DB_NAME |
| MinIO | 导入图片存储 | MINIO_ENDPOINT、MINIO_ACCESS_KEY、MINIO_SECRET_KEY、MINIO_BUCKET_NAME |
| MinerU API | PDF 转 Markdown | MINERU_BASE_URL、MINERU_API_TOKEN；MD 输入不需要 PDF 解析 |
| DashScope 重排接口 | 精排 | OPENAI_API_KEY、TEXT_RERANK_MODEL；即便聊天模型使用其他供应商，此重排实现仍调用 DashScope |
| Java 业务接口 | 真实场景及模型数据 | SPRINGBOOT_BASE_URL；该服务源码不包含在本仓库 |
| 百炼 MCP | 联网搜索 | MCP_DASHSCOPE_BASE_URL、OPENAI_API_KEY；只有选到联网工具时使用 |

Milvus、MongoDB、MinIO 和 Java 服务需要单独部署。这里未附完整 Docker 编排，也未把 Java 业务接口替换为假实现；缺少这些服务时请先使用 README 的本地检索示例。

## 操作顺序

1. 配好数据库、对象存储和模型服务后，分别启动 8002、8003 服务。
2. 在 `http://127.0.0.1:8002` 上传有权使用的 Markdown 或 PDF，等待导入任务完成。可用合成 Markdown 做首次验证；包含图片时需配置视觉模型和 MinIO，PDF 需配置 MinerU。
3. 请求查询，检查答案、任务状态及实际返回的文档。不要以 HTTP 200 作为答案可靠的唯一依据。

```powershell
$body = @{query='风机叶片检查记录需要包含什么？'; is_stream=$false} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8003/query -ContentType 'application/json; charset=utf-8' -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

如设置 `AGENT_API_KEY`，上述调用需增加 `-Headers @{'X-API-Key'='本地配置的值'}`。浏览器内置演示页目前未提供密钥输入。访问公网前应增加身份与任务归属控制，不能将共享 API Key 当作多用户隔离。

默认每个服务运行一个 worker，并持久化 `data/`。SQLite 事件日志目前没有自动清理策略，长期运行需要增加归档与容量监控。
