"""BGE-M3 混合向量化节点：为每个 Chunk 生成稠密+稀疏向量"""

import logging
from typing import List, Dict

from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.embedding_utils import generate_embeddings


class NodeBGEEmbedding(BaseNode):
    """
    BGE-M3 混合向量化节点：
    把每个切片的文本转成稠密向量(1024维) + 稀疏向量(BM25权重)
    """

    name = "node_bge_embedding"

    def process(self, state: ImportGraphState):
        # 步骤1：校验 chunks
        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks 不能为空")
        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks 类型必须为 list")

        # 步骤2：批量向量化
        output_data = self._step_2_generate_embeddings(chunks)

        # 步骤3：写回 state
        state["chunks"] = output_data
        return state

    def _step_2_generate_embeddings(self, chunks: List[Dict]) -> List[Dict]:
        """分批向量化，每批 8 个"""
        output_data = []
        batch_size = self.config.embedding_batch_size

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            # 拼接设施名 + 内容以增强语义
            input_texts = []
            for doc in batch:
                facility_name = doc.get("facility_name", "")
                content = doc.get("content", "")
                input_texts.append(f"{facility_name}\n{content}" if facility_name else content)

            # 批量向量化
            docs_embeddings = generate_embeddings(input_texts)

            # 给每个 chunk 加上向量
            for j, doc in enumerate(batch):
                item = doc.copy()
                item["dense_vector"] = docs_embeddings["dense"][j]
                item["sparse_vector"] = docs_embeddings["sparse"][j]
                output_data.append(item)

            self.logger.info(f"已向量化 {i + 1}-{min(i + len(batch), len(chunks))} 个切片")

        return output_data
