"""百炼 MCP WebSearch 配置"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class McpConfig:
    """百炼 MCP 联网搜索配置"""
    mcp_base_url: str = ""
    api_key: str = ""

    def __post_init__(self):
        if not self.mcp_base_url:
            self.mcp_base_url = os.getenv("MCP_DASHSCOPE_BASE_URL", "")
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")


mcp_config = McpConfig()
