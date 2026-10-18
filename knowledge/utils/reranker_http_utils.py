"""Reranker HTTP API 调用工具（DashScope TextReRank）"""

import logging
import requests
from knowledge.configs.reranker_config import reranker_config

logger = logging.getLogger("agent.reranker")


def rerank_documents(query: str, documents: list[str]) -> list[float]:
    """
    调用 DashScope TextReRank API 对文档进行重排序打分

    Args:
        query: 用户查询
        documents: 待排序的文档内容列表

    Returns:
        相关性分数列表（与输入顺序对应），值越高越相关

    Raises:
        RuntimeError: API 调用失败时抛出
    """
    if not documents:
        return []

    url = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    headers = {
        "Authorization": f"Bearer {reranker_config.text_rerank_api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": reranker_config.text_rerank_model,
        "input": {
            "query": query,
            "documents": documents,
        },
        "parameters": {
            "top_n": len(documents),
            "return_documents": False,
        },
    }

    try:
        resp = requests.post(url, json=body, headers=headers, timeout=30)
        resp.raise_for_status()
        result = resp.json()

        # API 返回的 results 按 score 降序排列，需要还原为输入顺序
        scored = result.get("output", {}).get("results", [])
        score_map = {item.get("index", -1): item.get("relevance_score", 0.0) for item in scored}
        return [score_map.get(i, 0.0) for i in range(len(documents))]
    except Exception as e:
        logger.error(f"Rerank API 调用失败: {e}")
        raise RuntimeError(f"Rerank 失败: {e}")
