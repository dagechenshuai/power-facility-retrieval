"""BGE-M3 嵌入模型配置"""

from dataclasses import dataclass, field
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class EmbeddingConfig:
    """BGE-M3 混合向量嵌入模型配置"""
    bge_m3_path: str = ""
    bge_m3: str = ""
    bge_device: str = "cpu"
    bge_fp16: bool = False
    embedding_batch_size: int = 8

    def __post_init__(self):
        if not self.bge_m3_path:
            self.bge_m3_path = os.getenv("BGE_M3_PATH", "")
        if not self.bge_m3:
            self.bge_m3 = os.getenv("BGE_M3", "BAAI/bge-m3")
        if not self.bge_device:
            self.bge_device = os.getenv("BGE_DEVICE", "cpu")
        if not self.bge_fp16:
            val = os.getenv("BGE_FP16", "False").lower()
            self.bge_fp16 = val in ("true", "1", "yes")
        self.embedding_batch_size = int(os.getenv("EMBEDDING_BATCH_SIZE", "8"))


embedding_config = EmbeddingConfig()
