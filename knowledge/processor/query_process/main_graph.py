"""知识库查询工作流 —— 4路并行检索 + RRF 融合 + Rerank 精排"""

from dotenv import load_dotenv
from langgraph.constants import END
from langgraph.graph import StateGraph

from knowledge.processor.query_process.nodes.node_facility_confirm import NodeFacilityConfirm
from knowledge.processor.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from knowledge.processor.query_process.nodes.node_search_hyde import NodeSearchEmbeddingHyde
from knowledge.processor.query_process.nodes.node_model_scene_search import NodeModelSceneSearch
from knowledge.processor.query_process.nodes.node_web_search_mcp import NodeWebSearchMcp
from knowledge.processor.query_process.nodes.node_rrf import NodeRrf
from knowledge.processor.query_process.nodes.node_rerank import NodeRerank
from knowledge.processor.query_process.nodes.node_answer_output import NodeAnswerOutput
from knowledge.processor.query_process.nodes.node_search_bm25 import NodeSearchBm25
from knowledge.processor.query_process.nodes.node_tool_router import NodeToolRouter
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.persistence import create_checkpointer

load_dotenv(override=True)


class KBQueryWorkflow:
    """知识库查询工作流 —— 4路并行搜索 + RRF + Rerank"""

    def __init__(self):
        self.workflow = StateGraph(QueryGraphState)
        self._init_nodes()
        self._register_nodes()
        self._setup_routes()
        self._compiled_app = None

    def _init_nodes(self):
        self.node_facility_confirm = NodeFacilityConfirm()
        self.node_search_embedding = NodeSearchEmbedding()
        self.node_search_hyde = NodeSearchEmbeddingHyde()
        self.node_backend_api = NodeModelSceneSearch()
        self.node_web_search = NodeWebSearchMcp()
        self.node_rrf = NodeRrf()
        self.node_rerank = NodeRerank()
        self.node_answer = NodeAnswerOutput()
        self.node_bm25 = NodeSearchBm25()
        self.node_tool_router = NodeToolRouter()

    def _register_nodes(self):
        self.workflow.add_node("node_facility_confirm", self.node_facility_confirm)
        self.workflow.add_node("node_search_embedding", self.node_search_embedding)
        self.workflow.add_node("node_search_hyde", self.node_search_hyde)
        self.workflow.add_node("node_backend_api", self.node_backend_api)
        self.workflow.add_node("node_web_search", self.node_web_search)
        self.workflow.add_node("node_rrf", self.node_rrf)
        self.workflow.add_node("node_rerank", self.node_rerank)
        self.workflow.add_node("node_answer_output", self.node_answer)
        self.workflow.add_node("node_search_bm25", self.node_bm25)
        self.workflow.add_node("node_tool_router", self.node_tool_router)

    def _route_after_confirm(self, state: QueryGraphState):
        """有预设 answer 直接输出，否则四路并行搜索"""
        if state.get("answer"):
            return "node_answer_output"
        return "node_tool_router"

    @staticmethod
    def _route_tools(state: QueryGraphState):
        mapping = {
            "vector_search": "node_search_embedding", "bm25_search": "node_search_bm25",
            "hyde_search": "node_search_hyde", "scene_search": "node_backend_api",
            "web_search": "node_web_search",
        }
        return [mapping[name] for name in state.get("selected_tools", []) if name in mapping]

    def _setup_routes(self):
        self.workflow.set_entry_point("node_facility_confirm")

        # 条件路由：有反问/拒识 → 直接输出，否则 → 并行搜索
        self.workflow.add_conditional_edges(
            "node_facility_confirm",
            self._route_after_confirm,
            {
                "node_answer_output": "node_answer_output",
                "node_tool_router": "node_tool_router",
            }
        )

        self.workflow.add_conditional_edges("node_tool_router", self._route_tools)

        # 四路搜索 → RRF → Rerank → 答案 → 结束
        self.workflow.add_edge("node_search_embedding", "node_rrf")
        self.workflow.add_edge("node_search_hyde", "node_rrf")
        self.workflow.add_edge("node_backend_api", "node_rrf")
        self.workflow.add_edge("node_web_search", "node_rrf")
        self.workflow.add_edge("node_search_bm25", "node_rrf")
        self.workflow.add_edge("node_rrf", "node_rerank")
        self.workflow.add_edge("node_rerank", "node_answer_output")
        self.workflow.add_edge("node_answer_output", END)

    def compile(self):
        if not self._compiled_app:
            self._compiled_app = self.workflow.compile(checkpointer=create_checkpointer())
        return self._compiled_app

    def run(self, initial_state: dict | None, stream: bool = False, thread_id: str | None = None):
        """启动查询流水线"""
        if not self._compiled_app:
            self.compile()
        config = {"configurable": {"thread_id": thread_id or "query-default"}}
        if stream:
            return self._compiled_app.stream(initial_state, config=config, stream_mode="values")
        else:
            return self._compiled_app.invoke(initial_state, config=config)
