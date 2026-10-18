"""查询流程状态类型定义"""

from typing import TypedDict, List, Annotated
import operator


def _keep_latest(x, y):
    """合并策略：并行节点冲突时取后一个值"""
    return y


class QueryGraphState(TypedDict, total=False):
    """查询流程图状态 —— 包含整个查询流程中传递的所有数据"""

    # 会话标识
    session_id: Annotated[str, _keep_latest]
    message_id: Annotated[str, _keep_latest]
    task_id: Annotated[str, _keep_latest]

    # 查询相关
    original_query: Annotated[str, _keep_latest]
    rewritten_query: Annotated[str, _keep_latest]

    # 设施/设备名称
    facility_names: Annotated[List[str], _keep_latest]

    # 各路检索结果
    embedding_chunks: Annotated[list, _keep_latest]       # 向量检索结果
    hyde_embedding_chunks: Annotated[list, _keep_latest]   # HyDE 检索结果
    bm25_chunks: Annotated[list, _keep_latest]
    backend_api_data: Annotated[list, _keep_latest]        # Spring Boot API 查询结果
    web_search_docs: Annotated[list, _keep_latest]          # MCP 网络搜索结果

    # HyDE 假设文档
    hyde_doc: Annotated[str, _keep_latest]

    # 融合与重排
    rrf_chunks: Annotated[list, _keep_latest]               # RRF 融合后
    reranked_docs: Annotated[list, _keep_latest]            # Rerank 精排后

    # 最终输出
    prompt: Annotated[str, _keep_latest]
    answer: Annotated[str, _keep_latest]

    # 会话历史
    history: Annotated[list, _keep_latest]

    # 控制标志
    is_stream: Annotated[bool, _keep_latest]
    selected_tools: Annotated[List[str], _keep_latest]
