"""
导入流程自定义异常类
"""


class ImportProcessError(Exception):
    """导入流程基础异常"""

    def __init__(self, message: str, node_name: str = "", cause: Exception = None):
        self.message = message
        self.node_name = node_name
        self.cause = cause
        super().__init__(message)

    def __str__(self):
        parts = [f"[{self.node_name}]" if self.node_name else "", self.message]
        if self.cause:
            parts.append(f"(原因: {self.cause})")
        return " ".join(filter(None, parts))


class StateFieldError(ImportProcessError):
    """状态字段缺失或类型错误"""

    def __init__(self, field_name: str = "", message: str = "", expected_type=None):
        msg = message or f"状态字段无效: {field_name}"
        if expected_type:
            msg += f"，期望类型: {expected_type}"
        super().__init__(msg)
        self.field_name = field_name


class FileProcessingError(ImportProcessError):
    """文件处理异常"""

    def __init__(self, message: str, file_path: str = ""):
        super().__init__(message)
        self.file_path = file_path


class ValidationError(ImportProcessError):
    """数据校验异常"""
    pass


class MilvusError(ImportProcessError):
    """Milvus 操作异常"""
    pass


class LLMError(ImportProcessError):
    """LLM 调用异常"""
    pass
