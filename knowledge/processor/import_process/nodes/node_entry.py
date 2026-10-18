"""入口节点：校验文件类型，设置路由标志"""

from pathlib import Path

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError, FileProcessingError, ValidationError
from knowledge.processor.import_process.state import ImportGraphState


class NodeEntry(BaseNode):
    """入口节点：判断文件类型 (PDF/MD)，设置后续路由"""

    name = "node_entry"

    def process(self, state: ImportGraphState):
        import_file_path = state.get("import_file_path")
        if not import_file_path:
            raise StateFieldError(field_name="import_file_path")

        import_file_path_obj = Path(import_file_path)
        if not import_file_path_obj.exists():
            raise FileProcessingError(
                message=f"文件不存在: {import_file_path}",
                file_path=import_file_path
            )

        # 判断类型
        suffix = import_file_path_obj.suffix.lower()
        if suffix == ".pdf":
            state["is_pdf_read_enabled"] = True
            state["pdf_path"] = import_file_path
            self.logger.info(f"识别为 PDF 文件: {import_file_path_obj.name}")
        elif suffix == ".md":
            state["is_md_read_enabled"] = True
            state["md_path"] = import_file_path
            state["md_content"] = import_file_path_obj.read_text(encoding="utf-8")
            self.logger.info(f"识别为 Markdown 文件: {import_file_path_obj.name}")
        else:
            raise ValidationError(message=f"不支持的文件格式: {suffix}（仅支持 .pdf 和 .md）")

        # 设置文件标题
        state["file_title"] = import_file_path_obj.stem
        if not state.get("file_dir"):
            state["file_dir"] = str(import_file_path_obj.parent)

        return state
