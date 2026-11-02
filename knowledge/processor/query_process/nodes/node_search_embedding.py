"""向量检索节点：BGE-M3 稠密+稀疏混合检索"""

import logging

from knowledge.configs.milvus_config import milvus_config
from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.tool_arguments import tool_query
from knowledge.utils.embedding_utils import generate_embeddings
from knowledge.utils.hybrid_search_utils import (
    create_hybrid_search_requests,
    hybrid_search,
)
from knowledge.utils.milvus_utils import get_milvus_client


class NodeSearchEmbedding(NodeBase):
    """节点2：基于改写查询 + 设施名过滤，执行 Milvus 向量混合检索"""

    name = "node_search_embedding"

    def process(self, state: QueryGraphState):
        try:
            query = tool_query(state, "vector_search")
            facility_names = state.get("facility_names", [])

            # 生成向量
            embeddings = generate_embeddings([query])
            dense_vec = embeddings["dense"][0]
            sparse_vec = embeddings["sparse"][0]

            # 构造过滤条件
            expr = None
            if facility_names:
                expr = f"facility_name in {facility_names}"
                self.logger.info(f"过滤条件: {expr}")
            else:
                self.logger.info("未指定设施名过滤，全库检索")

            # 混合搜索
            reqs = create_hybrid_search_requests(
                dense_vector=dense_vec, sparse_vector=sparse_vec, expr=expr, limit=10
            )

            self.logger.info("开始执行 Milvus 混合检索...")
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
                    "file_title",
                    "parent_id",
                    "parent_content",
                ],
            )

            search_results = res[0] if res else []
            self.logger.info(f"向量搜索完成，返回 {len(search_results)} 条结果")
            state["embedding_chunks"] = self._parents(search_results)
            return state
        except Exception as e:
            self.logger.exception(f"向量搜索失败: {e}")
            state["embedding_chunks"] = []
            return state

    @staticmethod
    def _parents(results):
        parents = {}
        for hit in results:
            entity = hit.get("entity", hit)
            parent_id = entity.get("parent_id") or str(entity.get("chunk_id", ""))
            parents.setdefault(
                parent_id,
                {
                    **entity,
                    "chunk_id": f"parent:{parent_id}",
                    "content": entity.get("parent_content")
                    or entity.get("content", ""),
                    "source": "parent_vector",
                },
            )
        return list(parents.values())
