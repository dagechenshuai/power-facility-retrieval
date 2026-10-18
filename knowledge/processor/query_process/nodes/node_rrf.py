"""RRF 倒数排名融合：将多路检索结果加权融合排序（支持 4 路）"""

import logging
from typing import List, Dict, Any, Tuple

from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState


class NodeRrf(NodeBase):
    """节点6：RRF 倒数排名融合 —— 将向量检索、HyDE检索、API检索、Web搜索 结果加权融合"""

    name = "node_rrf"

    def process(self, state: QueryGraphState):
        # 1. 提取各路结果
        embedding_list = self._extract_entities(state.get("embedding_chunks"))
        hyde_list = self._extract_entities(state.get("hyde_embedding_chunks"))
        bm25_list = self._extract_entities(state.get("bm25_chunks"))
        backend_list = self._format_backend_docs(state.get("backend_api_data", []))
        web_list = self._format_web_docs(state.get("web_search_docs", []))

        self.logger.info(
            f"RRF 输入: 向量={len(embedding_list)}, HyDE={len(hyde_list)}, "
            f"后端={len(backend_list)}, Web={len(web_list)}"
        )

        # 2. RRF 融合（4路，后端和 Web 权重较低）
        rrf_inputs = [
            (embedding_list, 1.0),
            (hyde_list, 1.0),
            (bm25_list, 1.0),
            (backend_list, 0.8),
            (web_list, 0.6),
        ]

        # 3. 融合
        rrf_merge_results = self._rrf_merge(rrf_inputs)

        # 4. 更新 state
        state["rrf_chunks"] = [doc for doc, _ in rrf_merge_results]
        self.logger.info(f"RRF 融合后共 {len(rrf_merge_results)} 条")

        return state

    def _extract_entities(self, chunks) -> list:
        """从 Milvus 检索结果中提取 entity"""
        if not chunks:
            return []
        result = []
        for doc in chunks:
            if isinstance(doc, dict):
                entity = doc.get("entity", doc)
                result.append(entity)
        return result

    def _format_backend_docs(self, backend_data) -> list:
        """格式化后端 API 结果为伪文档"""
        if not backend_data:
            return []
        result = []
        for i, item in enumerate(backend_data):
            if isinstance(item, dict):
                result.append({
                    "chunk_id": item.get("chunk_id", f"backend_{i}"),
                    "content": item.get("content", ""),
                    "title": item.get("title", ""),
                    "source": item.get("source", "backend"),
                })
        return result

    def _format_web_docs(self, web_docs) -> list:
        """格式化 Web 搜索结果"""
        if not web_docs:
            return []
        result = []
        for i, doc in enumerate(web_docs):
            result.append({
                "chunk_id": f"web_{i}",
                "content": doc.get("snippet", ""),
                "title": doc.get("title", ""),
                "source": "web",
                "url": doc.get("url", ""),
            })
        return result

    def _rrf_merge(
        self, rrf_inputs: List[Tuple[List, float]], k: int = 60, max_results: int = None
    ) -> List[Tuple[Dict, float]]:
        """
        RRF 核心公式：score = Σ weight / (k + rank)

        Args:
            rrf_inputs: [(文档列表, 权重), ...]
            k: 平滑常数
            max_results: 返回前 N 条
        """
        chunk_scores = {}
        chunk_data = {}

        for docs_list, weight in rrf_inputs:
            for rank, doc in enumerate(docs_list, start=1):
                chunk_id = doc.get("chunk_id", f"unknown_{rank}")
                chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0.0) + weight / (k + rank)
                chunk_data.setdefault(chunk_id, doc)

        sorted_results = sorted(
            [(chunk_data[cid], score) for cid, score in chunk_scores.items()],
            key=lambda x: x[1], reverse=True
        )

        return sorted_results[:max_results] if max_results else sorted_results
