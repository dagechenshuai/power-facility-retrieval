"""Rerank 精排节点：Cross-Encoder 打分 + 断崖检测动态截断"""

import logging
from typing import Dict, Any, List

from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.reranker_http_utils import rerank_documents

# 动态 TopK 参数
RERANK_MAX_TOPK = 10
RERANK_MIN_TOPK = 3
RERANK_GAP_ABS = 0.5      # 断崖绝对阈值
RERANK_GAP_RATIO = 0.25    # 断崖相对阈值


class NodeRerank(NodeBase):
    """节点7：Cross-Encoder 精排 + 断崖检测截断"""

    name = "node_rerank"

    def process(self, state: QueryGraphState):
        # 1. 合并 RRF 结果
        merged_docs = self._step_1_merge_docs(state)

        if not merged_docs:
            state["reranked_docs"] = []
            return state

        # 2. Rerank 精排
        ranked_docs = self._step_2_rerank(state, merged_docs)

        # 3. 断崖检测
        cutoff_docs = self._step_3_cliff_cutoff(ranked_docs)

        state["reranked_docs"] = cutoff_docs
        self.logger.info(f"Rerank 完成: {len(merged_docs)} 篇 → {len(cutoff_docs)} 篇")
        return state

    def _step_1_merge_docs(self, state: QueryGraphState) -> List[Dict]:
        """合并所有来源的文档"""
        docs = []
        for chunk in state.get("rrf_chunks", []):
            docs.append({
                "content": chunk.get("content", ""),
                "title": chunk.get("title", ""),
                "chunk_id": chunk.get("chunk_id", ""),
                "source": chunk.get("source", "local"),
                "url": chunk.get("url"),
            })
        return docs

    def _step_2_rerank(self, state: QueryGraphState, docs: List[Dict]) -> List[Dict]:
        """Cross-Encoder 精确打分"""
        try:
            query = state.get("rewritten_query", "")
            contents = [doc.get("content", "") for doc in docs]
            scores = rerank_documents(query, contents)

            scored = [{**doc, "score": score} for doc, score in zip(docs, scores)]
            scored.sort(key=lambda x: x.get("score", 0) or 0, reverse=True)
            return scored

        except Exception as e:
            self.logger.error(f"Rerank 失败: {e}")
            return [{**doc, "score": None} for doc in docs]

    def _step_3_cliff_cutoff(self, ranked_docs: List[Dict]) -> List[Dict]:
        """断崖检测：分数骤降处自动截断"""
        if not ranked_docs:
            return []

        upper = min(RERANK_MAX_TOPK, len(ranked_docs))
        lower = min(RERANK_MIN_TOPK, upper)
        cutoff = upper

        for i in range(lower - 1, upper - 1):
            cur = ranked_docs[i].get("score")
            nxt = ranked_docs[i + 1].get("score")
            if cur is None or nxt is None:
                continue

            abs_gap = cur - nxt
            rel_gap = abs_gap / (abs(cur) + 1e-6)

            if abs_gap >= RERANK_GAP_ABS or rel_gap >= RERANK_GAP_RATIO:
                cutoff = i + 1
                self.logger.info(f"断崖检测: 位置{cutoff}, abs_gap={abs_gap:.4f}, rel_gap={rel_gap:.4f}")
                break

        return ranked_docs[:cutoff]
