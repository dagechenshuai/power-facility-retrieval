"""
导入流程状态类型定义
"""

from typing import TypedDict, List
import copy


class ImportGraphState(TypedDict, total=False):
    """
    导入流程图状态 —— 包含整个导入流程中传递的所有数据
    """

    # ==================== 任务标识 ====================
    task_id: str

    # ==================== 控制标志 ====================
    is_md_read_enabled: bool
    is_pdf_read_enabled: bool

    # ==================== 路径信息 ====================
    import_file_path: str              # 导入文件的绝对路径
    file_dir: str                      # 工作目录（解压、生成中间文件）
    pdf_path: str                      # PDF 文件路径
    md_path: str                       # 转换后的 Markdown 文件路径

    # ==================== 文件信息 ====================
    file_title: str                    # 文件标题（不含扩展名）
    facility_name: str                 # 识别出的设施/设备名称

    # ==================== 处理中间数据 ====================
    md_content: str                    # Markdown 文档内容
    chunks: List                       # 文档切片列表


GRAPH_DEFAULT_STATE: ImportGraphState = {
    "task_id": "",
    "is_pdf_read_enabled": False,
    "is_md_read_enabled": False,
    "file_dir": "",
    "import_file_path": "",
    "pdf_path": "",
    "md_path": "",
    "file_title": "",
    "md_content": "",
    "chunks": [],
    "facility_name": "",
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建默认状态，支持覆盖指定字段"""
    state = copy.deepcopy(GRAPH_DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """获取默认状态副本（避免全局污染）"""
    return copy.deepcopy(GRAPH_DEFAULT_STATE)
