"""Spring Boot 后端 API 配置"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class SpringBootConfig:
    """Spring Boot 后端 REST API 连接配置"""
    base_url: str = ""
    timeout_seconds: int = 10

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.getenv("SPRINGBOOT_BASE_URL", "http://127.0.0.1:8081")


springboot_config = SpringBootConfig()
