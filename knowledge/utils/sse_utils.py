"""从持久事件日志独立读取 SSE，支持多连接和跨进程重放。"""

import asyncio
import json
import time

from knowledge.utils.persistence import append_event, get_task, list_events

POLL_SECONDS = 0.1
HEARTBEAT_SECONDS = 15


class SSEEvent:
    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"
    ERROR = "error"


def push_to_session(session_id: str, event: str, data: dict) -> None:
    """持久化事件；每个订阅者使用自己的游标读取。"""
    append_event(session_id, event, data)


async def sse_generator(session_id: str, request, after_sequence: int = 0):
    """按事件 ID 重放，独立订阅互不消费或删除对方的数据。"""
    cursor = after_sequence
    heartbeat = time.monotonic()
    while not await request.is_disconnected():
        messages = list_events(session_id, cursor)
        for index, message in enumerate(messages):
            cursor = message["sequence"]
            terminal = message["event"] in {SSEEvent.FINAL, SSEEvent.ERROR}
            task = get_task(session_id) if terminal else None
            # 恢复任务后，旧轮次的终止事件不能关闭当前订阅。
            if terminal and (
                index < len(messages) - 1 or (task and task["status"] == "processing")
            ):
                continue
            yield _format_event(message)
            if terminal:
                return
        # 客户端已消费最终事件后再次连接时，不无限等待。
        task = get_task(session_id)
        if not messages and task and task["status"] in {"completed", "failed"}:
            history = list_events(session_id)
            if (
                history
                and history[-1]["event"] in {SSEEvent.FINAL, SSEEvent.ERROR}
                and history[-1]["sequence"] <= cursor
            ):
                return
        if time.monotonic() - heartbeat >= HEARTBEAT_SECONDS:
            yield ": heartbeat\n\n"
            heartbeat = time.monotonic()
        await asyncio.sleep(POLL_SECONDS)


def _format_event(message: dict) -> str:
    return (
        f"id: {message['sequence']}\n"
        f"event: {message['event']}\n"
        f"data: {json.dumps(message['data'], ensure_ascii=False)}\n\n"
    )
