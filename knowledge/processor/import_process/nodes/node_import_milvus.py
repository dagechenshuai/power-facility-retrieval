"""导入 Milvus 节点：将向量化后的 Chunk 存入向量数据库"""

import logging
from typing import List, Dict

from pymilvus import DataType

from knowledge.configs.milvus_config import milvus_config
from knowledge.processor.import_process.base import BaseNode
from knowledge.processor.import_process.exceptions import StateFieldError, MilvusError
from knowledge.processor.import_process.state import ImportGraphState
from knowledge.utils.milvus_utils import get_milvus_client, escape_milvus_string
from knowledge.utils.bm25_store import replace_document


class NodeImportMilvus(BaseNode):
    """导入 Milvus 节点：创建集合 → 幂等清理 → 批量插入"""

    name = "node_import_milvus"

    def process(self, state: ImportGraphState):
        # 步骤1：校验输入
        chunks, vector_dim = self._step_1_check_input(state)

        # 步骤2：连接 Milvus，集合不存在则创建
        client = self._step_2_prepare_collection(vector_dim)

        # 步骤3：幂等清理旧数据
        self._step_3_clean_old_data(client, chunks)

        # 步骤4：批量插入
        updated_chunks = self._step_4_insert_data(client, chunks)
        replace_document(chunks[0].get("file_title", ""), updated_chunks)

        state["chunks"] = updated_chunks
        self.logger.info(f"导入 Milvus 完成，共计 {len(updated_chunks)} 条")
        return state

    def _step_1_check_input(self, state: ImportGraphState) -> tuple:
        chunks = state.get("chunks")
        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks 不能为空")

        first = chunks[0]
        if "dense_vector" not in first:
            raise StateFieldError(field_name="chunks", message="缺失 dense_vector 字段")
        if "sparse_vector" not in first:
            raise StateFieldError(field_name="chunks", message="缺失 sparse_vector 字段")

        return chunks, len(first["dense_vector"])

    def _step_2_prepare_collection(self, vector_dim: int):
        client = get_milvus_client()
        if not client:
            raise MilvusError("Milvus 连接失败")

        collection_name = milvus_config.chunks_collection
        if not client.has_collection(collection_name):
            self._create_collection(collection_name, client, vector_dim)

        return client

    def _create_collection(self, collection_name: str, client, vector_dim: int):
        """创建设施知识库 Collection"""
        schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="part", datatype=DataType.INT8)
        schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="facility_name", datatype=DataType.VARCHAR, max_length=200)
        schema.add_field(field_name="parent_id", datatype=DataType.VARCHAR, max_length=64)
        schema.add_field(field_name="parent_content", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dim)

        # 索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector", index_name="dense_vector_idx",
            index_type="AUTOINDEX", metric_type="COSINE"
        )
        index_params.add_index(
            field_name="sparse_vector", index_name="sparse_inverted_idx",
            index_type="SPARSE_INVERTED_INDEX", metric_type="IP",
            params={"inverted_index_algo": "DAAT_MAXSCORE", "normalize": True, "quantization": "none"}
        )

        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"Milvus Collection 创建完成: {collection_name}")

    def _step_3_clean_old_data(self, client, chunks):
        """按文件名幂等删除旧数据（同文档重新导入时避免重复）"""
        file_title = chunks[0].get("file_title", "")
        if file_title:
            safe_title = escape_milvus_string(file_title)
            try:
                client.delete(
                    collection_name=milvus_config.chunks_collection,
                    filter=f"file_title=='{safe_title}'"
                )
                self.logger.info(f"已清理旧数据: file_title={file_title}")
            except Exception as e:
                self.logger.error(f"清理旧数据失败: {e}")
                raise MilvusError(f"Milvus 数据删除失败: {e}")

    def _step_4_insert_data(self, client, chunks) -> List[Dict]:
        """批量插入并回填 chunk_id"""
        # 确保 part 字段存在
        for item in chunks:
            if "part" not in item:
                item["part"] = 0

        result = client.insert(
            collection_name=milvus_config.chunks_collection,
            data=chunks
        )
        inserted_count = result.get("insert_count", 0)
        self.logger.info(f"成功插入 {inserted_count} 条数据")

        # 回填 chunk_id
        inserted_ids = result.get("ids", [])
        if inserted_ids:
            for idx, item in enumerate(chunks):
                item["chunk_id"] = str(inserted_ids[idx])

        return chunks
