"""设施名确认节点：从用户问题中提取设施/设备/场景名称，改写查询"""

import json
import logging
from typing import Tuple, List, Dict

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.configs.milvus_config import milvus_config
from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.llm_utils import get_llm_client
from knowledge.utils.prompt_utils import load_prompt
from knowledge.utils.mongo_history_utils import (
    get_recent_messages, save_chat_message, update_message_facility_names
)
from knowledge.utils.embedding_utils import generate_embeddings
from knowledge.utils.hybrid_search_utils import create_hybrid_search_requests, hybrid_search
from knowledge.utils.milvus_utils import get_milvus_client


class NodeFacilityConfirm(NodeBase):
    """节点1：确认用户问题中的核心设施/设备/场景名称（含历史解析和 Milvus 对齐）"""

    name = "node_facility_confirm"

    def process(self, state: QueryGraphState):
        # 步骤1：校验参数
        session_id, original_query = self._step_1_validate(state)
        self.logger.info(f"步骤1: 会话={session_id}, 查询={original_query[:50]}...")

        # 步骤2：获取历史记录
        history = get_recent_messages(session_id)
        state["history"] = history

        # 步骤3：保存用户消息
        message_id = save_chat_message(session_id, "user", original_query)
        self.logger.info(f"步骤3: 用户消息已保存, id={message_id}")

        # 步骤4：LLM 提取设施名 + 改写问题
        extract_res = self._step_4_extract_info(original_query, history)
        facility_names = extract_res.get("facility_names", [])
        rewritten_query = extract_res.get("rewritten_query", original_query)
        state["rewritten_query"] = rewritten_query
        state["facility_names"] = facility_names

        # 步骤5+6：向量化设施名 → Milvus 搜索 → 置信度对齐
        align_result = {}
        if facility_names:
            query_results = self._step_5_vectorize_and_query(facility_names)
            align_result = self._step_6_align_names(query_results)
        else:
            self.logger.info("未提取到设施名，跳过向量对齐")

        # 步骤7：检查确认状态
        state = self._step_7_check_confirmation(state, align_result, history)

        # 步骤8：写入历史
        self._step_8_write_history(state, session_id, rewritten_query, message_id)
        return state

    # ==================== 步骤1 ====================

    def _step_1_validate(self, state: QueryGraphState) -> Tuple[str, str]:
        session_id = state.get("session_id")
        if not session_id:
            raise ValueError("核心参数 session_id 缺失")
        original_query = state.get("original_query")
        if not original_query:
            raise ValueError("核心参数 original_query 缺失")
        return session_id, original_query

    # ==================== 步骤4 ====================

    def _step_4_extract_info(self, query: str, history: list) -> Dict:
        try:
            client = get_llm_client()
            history_text = ""
            for msg in history:
                history_text += f"{msg.get('role', '')}: {msg.get('text', '')}\n"

            prompt = load_prompt("query_and_facility", history_text=history_text, query=query)

            messages = [
                SystemMessage(content="你是一个电力设施领域的智能助手，擅长理解用户意图和提取设施/设备/场景名称。"),
                HumanMessage(content=prompt)
            ]

            response = client.invoke(messages)
            content = response.content

            # 清洗可能包裹的 ```json ... ```
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()

            result = json.loads(content)
            self.logger.info(f"LLM 提取结果: {result}")

            if "facility_names" not in result:
                result["facility_names"] = []
            if "rewritten_query" not in result:
                result["rewritten_query"] = query

            return result
        except Exception as e:
            self.logger.error(f"LLM 设施名提取失败: {e}")
            return {"facility_names": [], "rewritten_query": query}

    # ==================== 步骤5 ====================

    def _step_5_vectorize_and_query(self, facility_names: List[str]) -> List[Dict]:
        self.logger.info(f"步骤5: 向量化并查询设施名: {facility_names}")
        results = []
        client = get_milvus_client()
        if not client:
            self.logger.error("Milvus 连接失败")
            return results

        collection_name = milvus_config.facility_name_collection
        embeddings = generate_embeddings(facility_names)

        for i in range(len(facility_names)):
            try:
                dense_vec = embeddings["dense"][i]
                sparse_vec = embeddings["sparse"][i]
                reqs = create_hybrid_search_requests(dense_vec, sparse_vec, limit=5)
                search_res = hybrid_search(
                    client, collection_name, reqs,
                    ranker_weights=(0.8, 0.2), limit=5,
                    output_fields=["facility_name"]
                )
                matches = []
                if search_res and len(search_res) > 0:
                    for hit in search_res[0]:
                        matches.append({
                            "facility_name": hit.get("entity", {}).get("facility_name"),
                            "score": hit.get("distance"),
                        })
                results.append({"extracted_name": facility_names[i], "matches": matches})
            except Exception as e:
                self.logger.error(f"查询 '{facility_names[i]}' 时出错: {e}")

        return results

    # ==================== 步骤6 ====================

    def _step_6_align_names(self, query_results: list) -> dict:
        confirmed = []
        options = []

        for res in query_results:
            extracted = (res.get("extracted_name", "") or "").strip()
            matches = res.get("matches", []) or []
            if not matches:
                continue

            high = [m for m in matches if m.get("score", 0) > 0.85]
            mid = [m for m in matches if m.get("score", 0) >= 0.6]

            if len(high) == 1:
                confirmed.append(high[0]["facility_name"])
            elif len(high) > 1:
                picked = None
                if extracted:
                    for m in high:
                        if m["facility_name"] == extracted:
                            picked = m
                            break
                if not picked:
                    picked = high[0]
                confirmed.append(picked["facility_name"])
            elif mid:
                for m in mid[:5]:
                    options.append(m["facility_name"])

        return {
            "confirmed_facility_names": list(set(confirmed)),
            "options": list(set(options))
        }

    # ==================== 步骤7 ====================

    def _step_7_check_confirmation(self, state, align_result, history):
        confirmed = align_result.get("confirmed_facility_names", [])
        options = align_result.get("options", [])

        if confirmed:
            # 更新历史消息的设施名
            ids_to_update = []
            for msg in history:
                if not msg.get("facility_names"):
                    mid = msg.get("_id")
                    if mid:
                        ids_to_update.append(str(mid))
            if ids_to_update:
                update_message_facility_names(ids_to_update, confirmed)

            state["facility_names"] = confirmed
            if state.get("answer", "").strip():
                state["answer"] = ""
            return state

        if options:
            opts_str = "、".join(options[:3])
            state["answer"] = f"您是想问以下哪个设施的相关信息：{opts_str}？请明确一下具体名称。"
            state["facility_names"] = []
            return state

        # 无结果 —— 不设置 answer，走正常搜索流程（全库搜索）
        state["facility_names"] = []
        return state

    # ==================== 步骤8 ====================

    def _step_8_write_history(self, state, session_id, rewritten_query, message_id):
        if state.get("answer"):
            save_chat_message(
                session_id=session_id, role="assistant", text=state["answer"],
                rewritten_query="", facility_names=state.get("facility_names", [])
            )
        save_chat_message(
            session_id=session_id, role="user", text=state["original_query"],
            rewritten_query=rewritten_query, facility_names=state.get("facility_names", []),
            message_id=message_id
        )
