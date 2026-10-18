"""LLM 模型配置"""

from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class LLMConfig:
    """大语言模型配置（兼容 OpenAI API 格式）"""
    base_url: str = ""
    api_key: str = ""
    vl_model: str = ""
    llm_model: str = ""
    llm_temperature: float = 0.1

    def __post_init__(self):
        if not self.base_url:
            self.base_url = os.getenv("OPENAI_API_BASE", "")
        if not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")
        if not self.vl_model:
            self.vl_model = os.getenv("VL_MODEL", "qwen3-vl-flash")
        if not self.llm_model:
            self.llm_model = os.getenv("LLM_DEFAULT_MODEL", "qwen-flash")
        if not self.llm_temperature:
            self.llm_temperature = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1"))


lm_config = LLMConfig()
