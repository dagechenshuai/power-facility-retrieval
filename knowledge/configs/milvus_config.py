"""Milvus 向量数据库配置"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class MilvusConfig:
    """Milvus 连接与集合配置"""
    milvus_url: str = ""
    chunks_collection: str = ""
    facility_name_collection: str = ""

    def __post_init__(self):
        if not self.milvus_url:
            self.milvus_url = os.getenv("MILVUS_URL", "")
        if not self.chunks_collection:
            self.chunks_collection = os.getenv("FACILITY_CHUNKS_COLLECTION", "uejava_facility_chunks")
        if not self.facility_name_collection:
            self.facility_name_collection = os.getenv("FACILITY_NAME_COLLECTION", "uejava_facility_names")


milvus_config = MilvusConfig()
