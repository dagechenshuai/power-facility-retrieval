"""设施名称识别节点：LLM 从文档中提取风场/变电站/设备名称"""

import json
import logging
from typing import Dict, List

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError, LLMError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.llm_utils import get_llm_client


class NodeFacilityNameRecognition(BaseNode):
    """
    设施名称识别节点：
    - 从文档前几个 Chunk 中提取电力设施/设备名称
    - 写入 state["faculty_name"] 和所有 chunk["facility_name"]
    """

    name = "node_facility_name"

    def process(self, state: ImportGraphState):
        # 步骤1：校验输入
        chunks = state.get("chunks")
        file_title = state.get("file_title")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks 不能为空")
        if not file_title:
            raise StateFieldError(field_name="file_title")

        # 步骤2：构建上下文（取前 K 个 chunk，限制总长度）
        k = self.config.facility_name_chunk_k
        max_chars = self.config.facility_name_chunk_size
        context_parts = []
        total_chars = 0
        for chunk in chunks[:k]:
            content = chunk.get("content", "")
            if total_chars + len(content) > max_chars:
                remaining = max_chars - total_chars
                context_parts.append(content[:remaining])
                break
            context_parts.append(content)
            total_chars += len(content)
        context = "\n\n".join(context_parts)

        # 步骤3：LLM 提取设施名称
        facility_name = self._step_3_extract_facility_name(file_title, context)

        # 步骤4：写入状态和所有 chunk
        state["facility_name"] = facility_name
        for chunk in chunks:
            chunk["facility_name"] = facility_name

        self.logger.info(f"设施名称识别结果: {facility_name}")
        return state

    def _step_3_extract_facility_name(self, file_title: str, context: str) -> str:
        """调用 LLM 提取设施名称"""
        prompt = f"""你是一个电力设施领域的专家。请从以下文档内容中提取核心设施/设备名称。

要求：
1. 返回设施的全称，包括品牌、型号和名称
2. 例如："金风科技 GW121-2500 风力发电机"、"松滋红光35KV变电站"、"黄梅岳窑110KV变电站"
3. 如果文中提到了多个设施，选择最主要的那个
4. 如果无法确定，使用文档标题作为设施名
5. 只返回设施名称，不要加其他文字

文档标题：{file_title}

文档内容：
{context[:3000]}

设施名称："""

        try:
            llm = get_llm_client()
            response = llm.invoke(prompt)
            name = response.content.strip()
            # 清理可能的多余输出
            if len(name) > 200:
                name = file_title
            return name or file_title
        except Exception as e:
            self.logger.error(f"LLM 设施名提取失败: {e}")
            return file_title
