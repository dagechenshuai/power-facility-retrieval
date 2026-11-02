"""由模型自主选择检索工具，失败时使用安全的本地检索降级。"""

from langchain_core.tools import tool

from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.llm_utils import get_llm_client


@tool
def vector_search(query: str) -> str:
    """检索电力设施知识库中的语义相关技术资料。"""
    return query


@tool
def bm25_search(query: str) -> str:
    """按型号、编号、规范名等精确关键词检索本地资料。"""
    return query


@tool
def hyde_search(query: str) -> str:
    """当用户表达与专业文档措辞可能差异较大时扩展语义检索。"""
    return query


@tool
def scene_search(query: str) -> str:
    """查询三维模型、已发布场景、坐标、地形和影像等实时业务数据。"""
    return query


@tool
def web_search(query: str) -> str:
    """仅在本地资料可能过时或用户明确询问外部最新信息时联网搜索。"""
    return query


class NodeToolRouter(NodeBase):
    name = "node_tool_router"
    tools = [vector_search, bm25_search, hyde_search, scene_search, web_search]

    def process(self, state: QueryGraphState):
        query = state.get("rewritten_query") or state.get("original_query", "")
        prompt = (
            "你是电力设施检索路由器。根据问题自主选择一个或多个工具。"
            "技术资料优先vector_search和bm25_search；措辞差异大时选hyde_search；"
            "模型、场景、坐标选scene_search；只有最新外部信息才选web_search。"
        )
        try:
            response = (
                get_llm_client()
                .bind_tools(self.tools, tool_choice="auto")
                .invoke([("system", prompt), ("human", query)])
            )
            allowed = {tool.name for tool in self.tools}
            queries = {}
            for call in response.tool_calls:
                name = call["name"]
                value = call.get("args", {}).get("query", query)
                if name in allowed and isinstance(value, str) and value.strip():
                    queries[name] = value.strip()[:4000]
            selected = list(queries)
        except Exception as exc:
            self.logger.warning("模型工具路由失败，降级为本地混合检索: %s", exc)
            selected = ["vector_search", "bm25_search"]
            queries = {name: query for name in selected}
        state["selected_tools"] = list(dict.fromkeys(selected)) or [
            "vector_search",
            "bm25_search",
        ]
        state["tool_queries"] = queries or {
            name: query for name in state["selected_tools"]
        }
        state["retrieval_round"] = state.get("retrieval_round", 0) + 1
        # 每轮只融合本轮工具返回的证据，旧证据保留在决策记录中。
        for key in (
            "embedding_chunks",
            "hyde_embedding_chunks",
            "bm25_chunks",
            "backend_api_data",
            "web_search_docs",
            "rrf_chunks",
            "reranked_docs",
        ):
            state[key] = []
        return state
