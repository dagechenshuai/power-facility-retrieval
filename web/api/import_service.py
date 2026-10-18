"""导入流程 Web API 服务 (端口 8002)"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent.parent.parent))

import os
import uuid
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, BackgroundTasks, File, UploadFile, HTTPException
from fastapi.responses import FileResponse

from knowledge.configs.minio_config import minio_config
from knowledge.processor.import_process.main_graph import KBImportWorkFlow
from knowledge.utils.minio_utils import get_minio_client
from knowledge.utils.task_utils import (
    add_running_task, add_done_task, update_task_status,
    get_task_status, get_done_task_list, get_running_task_list
)
from knowledge.utils.persistence import (
    check_state_store, get_task, list_tasks, mark_interrupted_tasks, upsert_task, update_task,
)
from web.api.runtime import configure_runtime

load_dotenv(override=True)
logger = logging.getLogger("import_service")

# FastAPI 应用
app = FastAPI(title="UEJAVA智能体-导入API", description="电力设施知识库导入流程 API")
configure_runtime(app)

MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
ALLOWED_UPLOAD_SUFFIXES = {".pdf", ".md"}


@app.on_event("startup")
async def recover_tasks_after_restart():
    interrupted = mark_interrupted_tasks("import")
    if interrupted:
        logger.warning("检测到 %s 个可恢复的导入任务", interrupted)
    if os.getenv("AUTO_RESUME_TASKS", "true").lower() not in {"true", "1", "yes"}:
        return
    for task in list_tasks("import", "interrupted"):
        payload = task["payload"]
        update_task(task["task_id"], status="processing", error="")
        asyncio.create_task(asyncio.to_thread(
            run_graph_task,
            task["task_id"], payload["file_dir"], payload["import_file_path"], True,
        ))


# ==================== 静态页面 ====================

@app.get("/")
@app.get("/import.html")
async def get_import_page():
    """导入管理页面"""
    current_dir = Path(__file__).absolute().parent.parent
    html_path = current_dir / "page" / "import.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="页面不存在")
    return FileResponse(html_path)


@app.get("/chat.html")
async def get_chat_page():
    """AI 聊天页面"""
    current_dir = Path(__file__).absolute().parent.parent
    html_path = current_dir / "page" / "chat.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="页面不存在")
    return FileResponse(html_path)


# ==================== 后台导入任务 ====================

def run_graph_task(task_id: str, file_dir: str, import_file_path: str, resume: bool = False):
    """LangGraph 全流程后台执行"""
    try:
        update_task_status(task_id, "processing")

        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }

        workflow = KBImportWorkFlow()
        graph_input = init_state
        if resume:
            config = {"configurable": {"thread_id": f"import:{task_id}"}}
            graph_input = None if workflow.compiled_graph.get_state(config).values else init_state
        for event in workflow.run(graph_input, stream=True,
                                  thread_id=f"import:{task_id}"):
            for node_name, _ in event.items():
                add_done_task(task_id, node_name)

        update_task_status(task_id, "completed")

    except Exception as e:
        update_task_status(task_id, "failed")
        update_task(task_id, error=str(e))
        logger.error(f"[{task_id}] 流程执行失败: {e}", exc_info=True)


# ==================== 文件上传 ====================

@app.post("/upload")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """上传 PDF/MD 文件并启动导入流程"""
    invalid_names = [
        Path(file.filename or "upload").name
        for file in files
        if Path(file.filename or "upload").suffix.lower() not in ALLOWED_UPLOAD_SUFFIXES
    ]
    if invalid_names:
        raise HTTPException(status_code=415, detail=f"不支持的文件类型: {', '.join(invalid_names)}")

    data_dir = os.path.join(
        os.getenv("DATA_BASED_ROOT_DIR", "./temp_data"),
        datetime.now().strftime("%Y%m%d")
    )
    task_ids = []

    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        safe_name = Path(file.filename or "upload").name
        logger.info(f"[{task_id}] 开始处理: {safe_name}")

        # 保存文件
        file_dir = os.path.join(data_dir, task_id)
        os.makedirs(file_dir, exist_ok=True)
        import_file_path = os.path.join(file_dir, safe_name)
        upsert_task(task_id, "import", "processing", {
            "file_dir": file_dir, "import_file_path": import_file_path
        })
        add_running_task(task_id, "upload_file")

        total_size = 0
        try:
            with open(import_file_path, "wb") as buffer:
                while chunk := await file.read(1024 * 1024):
                    total_size += len(chunk)
                    if total_size > MAX_UPLOAD_BYTES:
                        raise HTTPException(status_code=413, detail=f"文件超过大小限制: {safe_name}")
                    buffer.write(chunk)
        except Exception as exc:
            Path(import_file_path).unlink(missing_ok=True)
            update_task_status(task_id, "failed")
            update_task(task_id, error=str(exc))
            raise
        logger.info(f"[{task_id}] 文件已保存: {import_file_path}")

        # 上传到 MinIO（可选）
        try:
            minio_client = get_minio_client()
            if minio_client:
                object_name = f"facility_docs/{datetime.now().strftime('%Y%m%d')}/{safe_name}"
                minio_client.fput_object(
                    bucket_name=minio_config.bucket_name,
                    object_name=object_name,
                    file_path=import_file_path,
                    content_type=file.content_type or "application/octet-stream"
                )
                logger.info(f"[{task_id}] MinIO 上传成功")
        except Exception as e:
            logger.warning(f"[{task_id}] MinIO 上传失败: {e}")

        add_done_task(task_id, "upload_file")
        background_tasks.add_task(run_graph_task, task_id, file_dir, import_file_path)

    return {"code": 200, "message": f"文件上传成功, 共 {len(files)} 个", "task_ids": task_ids}


@app.post("/tasks/{task_id}/resume")
async def resume_import(task_id: str, background_tasks: BackgroundTasks):
    task = get_task(task_id)
    if not task or task["task_type"] != "import":
        raise HTTPException(status_code=404, detail="任务不存在")
    if task["status"] in {"processing", "completed"}:
        raise HTTPException(status_code=409, detail=f"任务状态不允许恢复: {task['status']}")
    payload = task["payload"]
    update_task(task_id, status="processing", error="")
    background_tasks.add_task(
        run_graph_task, task_id, payload["file_dir"], payload["import_file_path"], True
    )
    return {"task_id": task_id, "status": "processing"}


@app.get("/tasks/{task_id}")
async def get_import_task(task_id: str):
    task = get_task(task_id)
    if not task or task["task_type"] != "import":
        raise HTTPException(status_code=404, detail="任务不存在")
    return task


# ==================== 任务状态 ====================

@app.get("/status/{task_id}")
async def get_task_progress(task_id: str):
    """查询导入任务进度"""
    return {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    }


@app.get("/health")
async def health():
    return {"ok": True, "service": "import_service"}


@app.get("/ready")
async def ready():
    ready_state = check_state_store()
    return {"ok": ready_state, "checks": {"state_store": ready_state}}


# ==================== 启动 ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("导入服务启动中...")
    uvicorn.run(app=app, host="127.0.0.1", port=8002)
