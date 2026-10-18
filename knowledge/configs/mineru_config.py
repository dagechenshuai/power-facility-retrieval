"""MinerU PDF 解析 API 配置"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class MineruConfig:
    """MinerU PDF→Markdown 转换 API 配置"""
    base_url: str = ""
    api_token: str = ""

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.getenv("MINERU_BASE_URL", "")
        if not self.api_token:
            self.api_token = os.getenv("MINERU_API_TOKEN", "")


mineru_config = MineruConfig()
