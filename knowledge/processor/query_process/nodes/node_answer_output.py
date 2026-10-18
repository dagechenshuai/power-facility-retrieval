"""答案生成节点：组装 Prompt → LLM 生成最终回答 → 写入历史"""

import re
import logging
from typing import List, Dict, Tuple

from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.utils.llm_utils import get_llm_client
from knowledge.utils.mongo_history_utils import save_chat_message
from knowledge.utils.prompt_utils import load_prompt
from knowledge.utils.sse_utils import push_to_session, SSEEvent

MAX_CONTEXT_CHARS = 12000  # 上下文最大字符数


class NodeAnswerOutput(NodeBase):
    """节点8：组装 Prompt → LLM 生成最终答案 → 提取图片 → 写入 MongoDB"""

    name = "node_answer_output"

    def process(self, state: QueryGraphState):
        # 阶段一：检查是否已有 answer（来自反问/拒识）
        if not self._step_1_check_answer(state):
            # 阶段二：构建 Prompt
            prompt = self._step_2_construct_prompt(state)
            state["prompt"] = prompt

            # 阶段三：LLM 生成答案
            state = self._step_3_generate_response(state, prompt)

        # 提取图片
        image_urls = self._extract_images(state)

        # 阶段四：写入 MongoDB
        if state.get("answer"):
            self._step_4_write_history(state, image_urls)

        self.logger.info(f"答案生成完成，提取到 {len(image_urls)} 张图片")
        return state

    # ==================== 阶段一 ====================

    def _step_1_check_answer(self, state) -> bool:
        answer = state.get("answer")
        if answer:
            self.logger.info(f"已有预设答案: {answer[:50]}...")
            return True
        return False

    # ==================== 阶段二 ====================

    def _step_2_construct_prompt(self, state: QueryGraphState) -> str:
        question = state.get("rewritten_query") or state.get("original_query", "")
        facility_names = state.get("facility_names", [])

        # 格式化参考文档
        context_str, _ = self._format_reranked_docs(
            state.get("reranked_docs") or [], MAX_CONTEXT_CHARS
        )
        # 格式化历史对话
        history_str, _ = self._format_chat_history(
            state.get("history") or [], MAX_CONTEXT_CHARS
        )

        facility_str = ", ".join(facility_names) if facility_names else "未指定设施"

        prompt = load_prompt("answer",
            context=context_str or "无参考内容",
            history=history_str or "暂无历史对话",
            facility_names=facility_str,
            question=question
        )
        self.logger.info(f"Prompt 构建完成，长度: {len(prompt)}")
        return prompt

    def _format_reranked_docs(self, docs: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化重排序文档，带字符预算"""
        lines = []
        used = 0
        for idx, doc in enumerate(docs, start=1):
            content = doc.get("content", "")
            meta = [f"[{idx}]"]
            for field in ["source", "chunk_id", "url", "title"]:
                val = str(doc.get(field, "")).strip()
                if val:
                    meta.append(f"[{field}={val}]")
            score = doc.get("score")
            if score is not None:
                meta.append(f"[score={float(score):.4f}]")

            entry = " ".join(meta) + "\n" + content
            if used + len(entry) > char_budget:
                break
            lines.append(entry)
            used += len(entry) + 2
        return "\n\n".join(lines), char_budget - used

    def _format_chat_history(self, history: List[Dict], char_budget: int) -> Tuple[str, int]:
        """格式化对话历史"""
        lines = []
        used = 0
        role_map = {"user": "用户", "assistant": "助手"}
        for msg in history:
            role = msg.get("role", "")
            text = msg.get("text", "")
            if not text or role not in role_map:
                continue
            line = f"{role_map[role]}: {text}"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines), char_budget - used

    # ==================== 阶段三 ====================

    def _step_3_generate_response(self, state: QueryGraphState, prompt: str):
        self.logger.info("开始生成回答...")
        try:
            llm = get_llm_client()
            if state.get("is_stream"):
                final_text = ""
                for chunk in llm.stream(prompt):
                    delta = getattr(chunk, "content", "") or ""
                    final_text += delta
                    if delta:
                        push_to_session(
                            state.get("task_id", state.get("session_id", "default")),
                            SSEEvent.DELTA,
                            {"delta": delta},
                        )
                state["answer"] = final_text
            else:
                response = llm.invoke(prompt)
                state["answer"] = response.content

            self.logger.info(f"生成完成，长度: {len(state['answer'])}")
        except Exception as e:
            self.logger.exception(f"答案生成失败: {e}")
            raise RuntimeError("答案生成失败") from e
        return state

    # ==================== 图片提取 ====================

    def _extract_images(self, state: QueryGraphState) -> List[str]:
        """从参考文档中提取图片 URL"""
        images = []
        seen = set()
        docs = state.get("reranked_docs") or []
        if not docs:
            return images

        md_img_pattern = re.compile(r'!\[.*?\]\((.*?)\)')

        for doc in docs:
            url = (doc.get("url") or "").strip()
            if url and url.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp')):
                if url not in seen:
                    seen.add(url)
                    images.append(url)

            text = (doc.get("content") or "").strip()
            for img_url in md_img_pattern.findall(text):
                img_url = img_url.strip()
                if img_url and img_url not in seen:
                    seen.add(img_url)
                    images.append(img_url)

        return images

    # ==================== 阶段四 ====================

    def _step_4_write_history(self, state: QueryGraphState, image_urls=None):
        try:
            answer = (state.get("answer") or "").strip()
            if answer:
                save_chat_message(
                    session_id=state.get("session_id", "default"),
                    role="assistant",
                    text=answer,
                    rewritten_query="",
                    facility_names=state.get("facility_names") or [],
                    image_urls=image_urls,
                )
                self.logger.info("答案已写入 MongoDB 历史")
        except Exception as e:
            self.logger.error(f"写入历史失败: {e}")
