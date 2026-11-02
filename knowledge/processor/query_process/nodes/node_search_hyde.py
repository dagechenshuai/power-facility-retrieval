"""HyDE 检索节点：LLM 生成假设回答 → 向量检索"""

import logging

from knowledge.configs.milvus_config import milvus_config
from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.nodes.node_search_embedding import (
    NodeSearchEmbedding,
)
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.tool_arguments import tool_query
from knowledge.utils.embedding_utils import generate_embeddings
from knowledge.utils.hybrid_search_utils import (
    create_hybrid_search_requests,
    hybrid_search,
)
from knowledge.utils.llm_utils import get_llm_client
from knowledge.utils.milvus_utils import get_milvus_client
from knowledge.utils.prompt_utils import load_prompt


class NodeSearchEmbeddingHyde(NodeBase):
    """节点3：HyDE — LLM 先生成假设性理想答案，再用答案去向量库检索"""

    name = "node_search_embedding_hyde"

    def process(self, state: QueryGraphState):
        rewritten_query = tool_query(state, "hyde_search")
        facility_names = state.get("facility_names", [])

        try:
            # 步骤1：LLM 生成假设性文档
            hyde_doc = self._step_1_create_hyde_doc(rewritten_query)

            # 步骤2：问题 + 假设文档 拼接 → 向量 → 混合检索
            res = self._step_2_search_hyde(rewritten_query, hyde_doc, facility_names)

            state["hyde_embedding_chunks"] = res if res else []
            state["hyde_doc"] = hyde_doc
            return state

        except Exception as e:
            self.logger.exception(f"HyDE 检索失败: {e}")
            state["hyde_embedding_chunks"] = []
            return state

    def _step_1_create_hyde_doc(self, rewritten_query: str) -> str:
        """LLM 生成一段假设性的理想答案"""
        self.logger.info("步骤1: 生成假设性文档")
        llm = get_llm_client()
        hyde_prompt = load_prompt("hyde", rewritten_query=rewritten_query)
        hyde_doc = llm.invoke(hyde_prompt).content
        self.logger.info(f"假设文档已生成 ({len(hyde_doc)} 字)")
        return hyde_doc

    def _step_2_search_hyde(
        self, rewritten_query: str, hyde_doc: str, facility_names=None
    ):
        """问题 + 假设答案 → 向量 → Milvus 混合检索"""
        combined_text = f"{rewritten_query} {hyde_doc}"
        embeddings = generate_embeddings([combined_text])
        dense_vec = embeddings["dense"][0]
        sparse_vec = embeddings["sparse"][0]

        expr = None
        if facility_names:
            expr = f"facility_name in {facility_names}"

        reqs = create_hybrid_search_requests(
            dense_vector=dense_vec, sparse_vector=sparse_vec, expr=expr, limit=10
        )
        client = get_milvus_client()
        res = hybrid_search(
            client=client,
            collection_name=milvus_config.chunks_collection,
            reqs=reqs,
            ranker_weights=(0.8, 0.2),
            output_fields=[
                "chunk_id",
                "content",
                "facility_name",
                "title",
                "parent_id",
                "parent_content",
            ],
        )

        search_results = res[0] if res else []
        self.logger.info(f"HyDE 搜索完成，返回 {len(search_results)} 条")
        return NodeSearchEmbedding._parents(search_results)
