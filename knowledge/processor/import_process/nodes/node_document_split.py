"""文档切片节点：智能文档切片

将长文档切分为适合检索的 Chunk，采用「按标题初切 → 超长细切 → 过短合并」三层策略。
"""

import json
import re
import hashlib
from pathlib import Path
from typing import Tuple, List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState


class NodeDocumentSplit(BaseNode):
    """文档切分节点：按 Markdown 标题初切 → 超长细切 → 过短合并"""

    name = "node_document_split"

    def process(self, state: ImportGraphState):
        # 步骤1：获取输入并标准化
        content, file_title = self._step_1_get_inputs(state)

        # 步骤2：按标题初次切分
        sections, title_count, lines_count = self._step_2_split_by_titles(content, file_title)

        # 步骤3：无标题兜底
        sections = self._step_3_handle_no_title(content, sections, title_count, file_title)

        # 步骤4：精细化处理（长切短合）
        sections = self._step_4_refine_chunks(sections)

        # 步骤5：统计信息
        self._step_5_print_stats(lines_count, sections)

        # 步骤6：备份到 chunks.json
        self._step_6_backup(state, sections)

        state["chunks"] = sections
        return state

    # ==================== 步骤1 ====================

    def _step_1_get_inputs(self, state: ImportGraphState) -> Tuple[str, str]:
        file_title = state.get("file_title")
        md_path = state.get("md_path")
        md_content = state.get("md_content")

        if not file_title:
            raise StateFieldError(field_name="file_title")
        if not md_path:
            raise StateFieldError(field_name="md_path")
        if not md_content:
            raise StateFieldError(field_name="md_content")

        # 统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content, file_title

    # ==================== 步骤2 ====================

    def _step_2_split_by_titles(self, content: str, file_title: str) -> Tuple[List[Dict], int, int]:
        """按 Markdown 标题初次切分"""
        title_pattern = r'^\s*#{1,6}\s+.+'
        lines = content.split("\n")
        sections = []
        title_count = 0
        current_title = ""
        current_lines = []
        in_code_block = False
        code_block_start_marker = None

        def _flush():
            if not current_lines:
                return
            sections.append({
                "title": current_title,
                "content": "\n".join(current_lines),
                "file_title": file_title,
            })

        for line in lines:
            stripped = line.strip()

            # 代码块边界识别
            code_marker = re.match(r'^(`{3,}|~{3,})$', stripped)
            if code_marker:
                marker = code_marker.group(1)
                if not in_code_block:
                    in_code_block = True
                    code_block_start_marker = marker
                elif stripped == code_block_start_marker:
                    in_code_block = False
                    code_block_start_marker = None
                current_lines.append(line)
                continue

            # 识别标题
            if (not in_code_block) and re.match(title_pattern, line):
                _flush()
                current_title = stripped
                current_lines = [current_title]
                title_count += 1
            else:
                current_lines.append(line)

        _flush()
        self.logger.info(f"按标题初切完成：{len(sections)} 个章节，{title_count} 个标题，{len(lines)} 行")
        return sections, title_count, len(lines)

    # ==================== 步骤3 ====================

    def _step_3_handle_no_title(self, content: str, sections: List[Dict],
                                  title_count: int, file_title: str) -> List[Dict]:
        if title_count == 0:
            self.logger.warning(f"未识别到标题，将全文作为单个章节")
            return [{"title": "全文", "content": content, "file_title": file_title}]
        return sections

    # ==================== 步骤4 ====================

    def _step_4_refine_chunks(self, sections: List[Dict]) -> List[Dict]:
        """长切短合：超长章节二次切分 → 过短章节合并 → 父标题兜底"""
        max_len = self.config.max_content_length  # 2000
        min_len = self.config.min_content_length  # 500

        # 阶段1：超长切分
        refined = []
        for sec in sections:
            identity = f"{sec.get('file_title', '')}\n{sec.get('title', '')}\n{sec.get('content', '')}"
            sec["parent_id"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            sec["parent_content"] = sec.get("content", "")
            refined.extend(self._split_long_section(sec, max_len))
        self.logger.info(f"超长切分后: {len(refined)} 个 Chunk")

        # 阶段2：过短合并
        final = self._merge_short_sections(refined, min_len)
        self.logger.info(f"过短合并后: {len(final)} 个 Chunk")

        # 阶段3：父标题兜底
        for sec in final:
            if not sec.get("parent_title"):
                sec["parent_title"] = sec.get("title") or "无标题"

        return final

    def _split_long_section(self, section: Dict, max_len: int) -> List[Dict]:
        """超长章节递归切分"""
        content = section.get("content", "")
        if len(content) <= max_len:
            return [section]

        title = section.get("title", "")
        prefix = f"{title}\n\n" if title else ""
        available_len = max_len - len(prefix)

        if available_len <= 0:
            self.logger.warning(f"标题过长无法切分: {title[:20]}...")
            return [section]

        # 去重标题
        body = content
        if title and body.lstrip().startswith(title):
            body = body[body.find(title) + len(title):].lstrip()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=available_len,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", "！", "？", "；", ".", "!", "?", ";", " "],
        )

        sub_sections = []
        for idx, chunk in enumerate(splitter.split_text(body), start=1):
            text = chunk.strip()
            if not text:
                continue
            full_text = (prefix + text).strip()
            sub_sections.append({
                "title": f"{title}-{idx}" if title else f"chunk-{idx}",
                "content": full_text,
                "parent_title": title,
                "part": idx,
                "file_title": section.get("file_title"),
                "parent_id": section.get("parent_id"),
                "parent_content": section.get("parent_content"),
            })

        return sub_sections

    def _merge_short_sections(self, sections: List[Dict], min_len: int) -> List[Dict]:
        """合并同一父标题下的过短 Chunk"""
        if not sections:
            return []

        merged = []
        current = None

        for sec in sections:
            if current is None:
                current = sec
                continue

            is_short = len(current["content"]) < min_len
            is_same_parent = current.get("parent_id") == sec.get("parent_id")

            if is_short and is_same_parent:
                parent = sec.get("parent_title", "")
                next_content = sec["content"]
                if parent and next_content.startswith(parent):
                    next_content = next_content[len(parent):].lstrip()
                current["content"] += "\n\n" + next_content
                if "part" in sec:
                    current["part"] = sec["part"]
            else:
                merged.append(current)
                current = sec

        if current is not None:
            merged.append(current)

        return merged

    # ==================== 步骤5 ====================

    def _step_5_print_stats(self, lines_count: int, sections: List[Dict]) -> None:
        self.logger.info("-" * 50)
        self.logger.info(f"原始文本行数: {lines_count}")
        self.logger.info(f"最终 Chunk 数量: {len(sections)}")
        if sections:
            preview = sections[0]["content"][:100].replace("\n", "\\n")
            self.logger.info(f"首个 Chunk 预览: {preview}")

    # ==================== 步骤6 ====================

    def _step_6_backup(self, state: ImportGraphState, sections: List[Dict]) -> None:
        try:
            backup_path = Path(state["md_path"]).parent / "chunks.json"
            with open(backup_path, "w", encoding="utf-8") as f:
                json.dump(sections, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Chunk 备份: {backup_path}")
        except Exception as e:
            self.logger.error(f"Chunk 备份失败: {e}")
