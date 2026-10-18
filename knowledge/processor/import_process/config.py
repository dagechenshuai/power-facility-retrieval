"""
导入流程配置管理

集中管理所有导入配置项，支持环境变量覆盖
"""

from dataclasses import dataclass, field
from typing import Set, Optional
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class ImportConfig:
    """导入流程配置"""

    # ==================== 文档处理配置 ====================
    max_content_length: int = 2000       # 切片最大长度
    img_content_length: int = 200        # 图片上下文最大长度
    min_content_length: int = 500        # 合并短内容的最小长度
    overlap_sentences: int = 1           # 句子级切分重叠句数
    facility_name_chunk_k: int = 3       # 设施名识别时使用的切片数量
    facility_name_chunk_size: int = 2500 # 设施名识别时使用的切片内容长度

    image_extensions: Set[str] = field(
        default_factory=lambda: {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    )

    # ==================== 向量配置 ====================
    embedding_dim: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "1024"))
    )
    embedding_batch_size: int = 8

    # ==================== 速率限制 ====================
    requests_per_minute: int = 15
    temp_dir: str = field(default_factory=lambda: os.getenv("AGENT_TEMP_DIR", "./temp-files"))


# ==================== 全局单例 ====================
_config: Optional[ImportConfig] = None


def get_config() -> ImportConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        _config = ImportConfig()
    return _config
