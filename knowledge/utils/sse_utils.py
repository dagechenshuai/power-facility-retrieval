"""SSE（Server-Sent Events）流式推送工具"""

import asyncio
import json
from typing import Dict

from knowledge.utils.persistence import append_event, list_events


class SSEEvent:
    """SSE 事件类型常量"""
    READY = "ready"
    PROGRESS = "progress"
    DELTA = "delta"
    FINAL = "final"
    ERROR = "error"


# 队列与所属事件循环必须一起保存，后台线程通过 call_soon_threadsafe 投递。
_sse_queues: Dict[str, tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = {}


def create_sse_queue(session_id: str):
    """为指定会话创建 SSE 消息队列"""
    if session_id not in _sse_queues:
        _sse_queues[session_id] = (asyncio.get_running_loop(), asyncio.Queue())


def push_to_session(session_id: str, event: str, data: dict):
    """向指定会话推送 SSE 事件"""
    sequence = append_event(session_id, event, data)
    channel = _sse_queues.get(session_id)
    if channel:
        loop, queue = channel
        message = {"sequence": sequence, "event": event, "data": data}
        try:
            loop.call_soon_threadsafe(queue.put_nowait, message)
        except RuntimeError:
            # 服务关闭期间事件仍已落盘，重启后可通过 Last-Event-ID 补发。
            pass


async def sse_generator(session_id: str, request, after_sequence: int = 0):
    """
    SSE 事件生成器（用于 FastAPI StreamingResponse）

    使用方式:
        @app.get("/stream/{session_id}")
        async def stream(session_id: str, request: Request):
            return StreamingResponse(
                sse_generator(session_id, request),
                media_type="text/event-stream"
            )
    """
    # 先补发断线期间已经持久化的事件。
    persisted_events = list_events(session_id, after_sequence)
    last_sequence = after_sequence
    for msg in persisted_events:
        last_sequence = msg["sequence"]
        yield _format_event(msg)
        if msg["event"] in (SSEEvent.FINAL, SSEEvent.ERROR):
            return

    # 恢复后可能尚无进程内队列，创建队列继续接收后续事件。
    create_sse_queue(session_id)
    _, queue = _sse_queues[session_id]

    # 发送就绪信号
    yield f"event: {SSEEvent.READY}\ndata: {json.dumps({'status': 'ready'})}\n\n"

    try:
        while True:
            # 检查客户端是否断开
            if await request.is_disconnected():
                break

            try:
                msg = await asyncio.wait_for(queue.get(), timeout=15.0)
            except asyncio.TimeoutError:
                # 心跳
                yield ": heartbeat\n\n"
                continue

            event = msg.get("event", "message")
            if msg.get("sequence", 0) <= last_sequence:
                continue
            last_sequence = msg.get("sequence", last_sequence)

            yield _format_event(msg)

            if event in (SSEEvent.FINAL, SSEEvent.ERROR):
                break
    finally:
        # 清理队列
        _sse_queues.pop(session_id, None)


def _format_event(msg: dict) -> str:
    return (
        f"id: {msg.get('sequence', '')}\n"
        f"event: {msg.get('event', 'message')}\n"
        f"data: {json.dumps(msg.get('data', {}), ensure_ascii=False)}\n\n"
    )
