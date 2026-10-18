"""传统 BM25 关键词检索节点。"""

from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.bm25_store import search_bm25


class NodeSearchBm25(NodeBase):
    name = "node_search_bm25"

    def process(self, state: QueryGraphState):
        state["bm25_chunks"] = search_bm25(
            state.get("rewritten_query", ""), state.get("facility_names", []), limit=10
        )
        return state
