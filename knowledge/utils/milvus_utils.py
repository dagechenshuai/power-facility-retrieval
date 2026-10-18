"""Milvus 客户端与工具函数"""

from pymilvus import MilvusClient
from knowledge.configs.milvus_config import milvus_config

_client = None


def get_milvus_client():
    """获取 Milvus 客户端单例"""
    global _client
    if _client is None:
        _client = MilvusClient(uri=milvus_config.milvus_url)
    return _client


def escape_milvus_string(value: str) -> str:
    """转义 Milvus 过滤表达式中的特殊字符"""
    if value is None:
        return ""
    s = str(value)
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    s = s.replace("\r", " ").replace("\n", " ")
    return s
