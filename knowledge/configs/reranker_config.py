"""Reranker 重排序模型配置"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class RerankerConfig:
    """DashScope TextReRank 重排序配置"""
    text_rerank_api_key: str = ""
    text_rerank_model: str = ""
    text_rerank_instruct: str = ""

    def __post_init__(self):
        if not self.text_rerank_api_key:
            self.text_rerank_api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.text_rerank_model:
            self.text_rerank_model = os.getenv("TEXT_RERANK_MODEL", "qwen3-rerank")
        if not self.text_rerank_instruct:
            self.text_rerank_instruct = os.getenv(
                "TEXT_RERANK_INSTRUCT",
                "针对给定的查询，检索能够解答该查询的相关段落"
            )


reranker_config = RerankerConfig()
