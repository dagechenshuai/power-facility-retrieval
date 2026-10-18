"""查询流程节点基类"""

from abc import ABC, abstractmethod
from typing import TypeVar
import logging

T = TypeVar("T")


class NodeBase(ABC):
    """查询流程节点基类 —— 每个查询节点继承此类并实现 process 方法"""

    name: str = "base_node"

    def __init__(self):
        self.logger = logging.getLogger(f"query.{self.name}")

    def __call__(self, state: T) -> T:
        """节点执行入口（LangGraph 调用）"""
        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            result = self.process(state)
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}", exc_info=True)
            raise

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑"""
        ...
