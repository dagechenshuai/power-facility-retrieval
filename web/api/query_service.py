"""查询流程 Web API 服务 (端口 8003)"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

import asyncio
import logging
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from knowledge.processor.query_process.main_graph import KBQueryWorkflow
from knowledge.utils.mongo_history_utils import clear_history, get_recent_messages
from knowledge.utils.persistence import (
    check_state_store,
    get_task,
    list_tasks,
    mark_interrupted_tasks,
    update_task,
    upsert_task,
)
from knowledge.utils.sse_utils import SSEEvent, push_to_session, sse_generator
from knowledge.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    get_task_result,
    set_task_result,
    update_task_status,
)
from web.api.runtime import configure_runtime

load_dotenv(override=True)
logger = logging.getLogger("query_service")

# FastAPI 应用
app = FastAPI(title="UEJAVA智能体-查询API", description="电力设施知识库智能问答 API")
configure_runtime(app)


@app.on_event("startup")
async def recover_tasks_after_restart():
    interrupted = mark_interrupted_tasks("query")
    if interrupted:
        logger.warning("检测到 %s 个可恢复的查询任务", interrupted)
    if os.getenv("AUTO_RESUME_TASKS", "true").lower() not in {"true", "1", "yes"}:
        return
    for task in list_tasks("query", "interrupted"):
        payload = task["payload"]
        update_task(task["task_id"], status=TASK_STATUS_PROCESSING, error="")
        asyncio.create_task(
            asyncio.to_thread(
                run_query_graph,
                task["task_id"],
                payload["session_id"],
                payload["query"],
                True,
                True,
            )
        )


# ==================== 数据模型 ====================


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000, description="用户查询内容")
    session_id: str | None = Field(None, description="会话ID（可选，不传则自动创建）")
    is_stream: bool = Field(False, description="是否使用 SSE 流式输出")


# ==================== 静态页面 ====================


@app.get("/")
@app.get("/chat.html")
async def chat():
    """聊天页面"""
    current_dir = Path(__file__).absolute().parent.parent
    html_path = current_dir / "page" / "chat.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="页面不存在")
    return FileResponse(html_path)


# ==================== 后台执行检索流程 ====================


def run_query_graph(
    task_id: str,
    session_id: str,
    user_query: str,
    is_stream: bool = True,
    resume: bool = False,
):
    """后台执行检索全流程"""
    logger.info(f"开始检索: [{session_id}] {user_query}")
    init_state = (
        None
        if resume
        else {
            "original_query": user_query,
            "session_id": session_id,
            "task_id": task_id,
            "answer": "",
            "is_stream": is_stream,
        }
    )
    try:
        update_task_status(task_id, TASK_STATUS_PROCESSING, False)
        workflow = KBQueryWorkflow()
        final_state = {}
        if resume:
            graph = workflow.compile()
            config = {"configurable": {"thread_id": f"query:{task_id}"}}
            final_state = graph.get_state(config).values or {}
            if not final_state:
                init_state = {
                    "original_query": user_query,
                    "session_id": session_id,
                    "task_id": task_id,
                    "answer": "",
                    "is_stream": is_stream,
                }
        answer = final_state.get("answer")
        for event in workflow.run(
            init_state, stream=True, thread_id=f"query:{task_id}"
        ):
            if isinstance(event, dict):
                final_state = event
                if event.get("answer"):
                    answer = event["answer"]

        if not answer or not answer.strip():
            raise RuntimeError("模型未生成有效答案")
        final_answer = answer
        set_task_result(task_id, "answer", final_answer)
        for key in ("result_status", "decision_trace", "reranked_docs"):
            set_task_result(task_id, key, final_state.get(key))
        update_task_status(task_id, TASK_STATUS_COMPLETED, False)
        if is_stream:
            push_to_session(
                task_id,
                SSEEvent.FINAL,
                {
                    "answer": final_answer,
                    "status": "completed",
                    "result_status": final_state.get("result_status"),
                },
            )

    except Exception as e:
        logger.error(f"检索异常: {e}", exc_info=True)
        update_task_status(task_id, TASK_STATUS_FAILED, False)
        update_task(task_id, error=str(e))
        if is_stream:
            push_to_session(task_id, SSEEvent.ERROR, {"error": str(e)})


# ==================== 查询接口 ====================


@app.post("/query")
async def query(background_tasks: BackgroundTasks, request: QueryRequest):
    """
    智能问答接口

    - is_stream=false: 同步返回完整答案
    - is_stream=true: 异步处理，通过 SSE 流式返回
    """
    session_id = request.session_id or str(uuid.uuid4())
    task_id = str(uuid.uuid4())
    is_stream = request.is_stream

    upsert_task(
        task_id,
        "query",
        TASK_STATUS_PROCESSING,
        {"session_id": session_id, "query": request.query, "is_stream": is_stream},
    )

    if is_stream:
        background_tasks.add_task(
            run_query_graph, task_id, session_id, request.query, is_stream
        )
        return {
            "message": "结果正在处理中...",
            "session_id": session_id,
            "task_id": task_id,
        }
    else:
        await asyncio.to_thread(
            run_query_graph, task_id, session_id, request.query, is_stream
        )
        task = get_task(task_id)
        if not task or task["status"] != TASK_STATUS_COMPLETED:
            return JSONResponse(
                status_code=502,
                content={
                    "message": "处理失败",
                    "status": "failed",
                    "session_id": session_id,
                    "task_id": task_id,
                    "error": "查询执行失败，请查看任务状态或服务日志",
                },
            )
        answer = get_task_result(task_id, "answer", "")
        return {
            "status": "completed",
            "message": "处理完成",
            "session_id": session_id,
            "task_id": task_id,
            "answer": answer,
            "result_status": get_task_result(task_id, "result_status"),
            "decision_trace": get_task_result(task_id, "decision_trace", []),
        }


# ==================== SSE 流式接口 ====================


@app.get("/stream/{session_id}")
async def stream(session_id: str, request: Request):
    """SSE 实时推送端点"""
    return StreamingResponse(
        sse_generator(
            session_id, request, int(request.headers.get("last-event-id", "0") or 0)
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/tasks/{task_id}/resume")
async def resume_task(task_id: str, background_tasks: BackgroundTasks):
    task = get_task(task_id)
    if not task or task["task_type"] != "query":
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] in {TASK_STATUS_PROCESSING, TASK_STATUS_COMPLETED}:
        raise HTTPException(
            status_code=409, detail=f"任务状态不允许恢复: {task['status']}"
        )
    update_task(task_id, status=TASK_STATUS_PROCESSING, error="")
    payload = task["payload"]
    background_tasks.add_task(
        run_query_graph, task_id, payload["session_id"], payload["query"], True, True
    )
    return {"task_id": task_id, "status": "processing"}


@app.get("/tasks/{task_id}")
async def get_query_task(task_id: str):
    task = get_task(task_id)
    if not task or task["task_type"] != "query":
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


# ==================== 会话历史 ====================


@app.get("/history/{session_id}")
async def history(session_id: str, limit: int = 50):
    """获取对话历史"""
    try:
        records = get_recent_messages(session_id, limit=limit)
        items = []
        for r in records:
            items.append(
                {
                    "_id": str(r.get("_id")) if r.get("_id") else "",
                    "session_id": r.get("session_id", ""),
                    "role": r.get("role", ""),
                    "text": r.get("text", ""),
                    "rewritten_query": r.get("rewritten_query", ""),
                    "facility_names": r.get("facility_names", []),
                    "ts": str(r.get("ts")) if r.get("ts") else "",
                }
            )
        return {"session_id": session_id, "items": items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询历史失败: {e}")


@app.delete("/history/{session_id}")
async def clear_chat_history(session_id: str):
    """清空对话历史"""
    count = clear_history(session_id)
    return {"message": "历史会话已清空", "deleted_count": count}


@app.get("/health")
async def health():
    return {"ok": True, "service": "query_service"}


@app.get("/ready")
async def ready():
    ready_state = check_state_store()
    return {"ok": ready_state, "checks": {"state_store": ready_state}}


# ==================== 启动 ====================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info("查询服务启动中...")
    uvicorn.run(app=app, host="127.0.0.1", port=8003)
