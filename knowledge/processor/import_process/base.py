"""
导入流程节点基类

定义统一的节点接口规范，提供日志、任务追踪和异常处理。
所有导入节点必须继承此基类并实现 process 方法。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

from knowledge.processor.import_process.config import ImportConfig, get_config
from knowledge.processor.import_process.exceptions import ImportProcessError

T = TypeVar("T")


class BaseNode(ABC):
    """
    导入流程节点基类

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        node = MyNode()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"

    def __init__(self, config: Optional[ImportConfig] = None):
        self.config = config or get_config()
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口（LangGraph 调用）
        提供统一的日志、任务追踪和异常处理
        """
        try:
            self.logger.info(f"--- {self.name} 开始 ---")
            result = self.process(state)
            self.logger.info(f"--- {self.name} 完成 ---")
            return result
        except Exception as e:
            self.logger.error(f"{self.name} 执行失败: {e}")
            raise ImportProcessError(
                message=str(e),
                node_name=self.name,
                cause=e
            )

    @abstractmethod
    def process(self, state: T) -> T:
        """节点核心处理逻辑，子类必须实现"""
        ...

    def log_step(self, step_name: str, message: str = ""):
        """记录步骤日志"""
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


def setup_logging(level: int = logging.INFO):
    """配置导入流程日志"""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
