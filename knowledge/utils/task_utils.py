"""任务状态追踪工具（导入流程进度）"""

from knowledge.utils.sse_utils import push_to_session, SSEEvent
from knowledge.utils.persistence import get_task, update_task, upsert_task

# 任务状态常量
TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_INTERRUPTED = "interrupted"

# 全局任务字典: {task_id: {"status": str, "done": [...], "running": [...], "results": {}}}
_tasks: dict = {}


def _ensure_task(task_id: str):
    """确保任务记录存在"""
    if task_id in _tasks:
        return
    _tasks[task_id] = {
        "status": TASK_STATUS_PENDING,
        "done": [],
        "running": [],
        "results": {},
    }
    persisted = get_task(task_id)
    if not persisted:
        return
    _tasks[task_id]["status"] = persisted["status"]
    persisted_result = persisted["result"]
    _tasks[task_id]["done"] = persisted_result.get("done", [])
    _tasks[task_id]["running"] = persisted_result.get("running", [])
    _tasks[task_id]["results"] = persisted_result.get("values", {})


def _persist_task_progress(task_id: str) -> None:
    task = _tasks[task_id]
    update_task(task_id, result={
        "done": task["done"],
        "running": task["running"],
        "values": task["results"],
    })


def add_running_task(task_id: str, node_name: str):
    """标记节点正在运行"""
    _ensure_task(task_id)
    if node_name not in _tasks[task_id]["running"]:
        _tasks[task_id]["running"].append(node_name)
        _persist_task_progress(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False):
    """将节点从运行中移到已完成"""
    _ensure_task(task_id)
    task = _tasks[task_id]
    if node_name in task["running"]:
        task["running"].remove(node_name)
    if node_name not in task["done"]:
        task["done"].append(node_name)
    _persist_task_progress(task_id)

    if is_stream:
        push_to_session(task_id, SSEEvent.PROGRESS, {
            "node": node_name,
            "done": task["done"],
            "running": task["running"],
        })


def update_task_status(task_id: str, status: str, is_stream: bool = False):
    """更新任务整体状态"""
    _ensure_task(task_id)
    _tasks[task_id]["status"] = status
    if get_task(task_id):
        update_task(task_id, status=status)
    else:
        upsert_task(task_id, "unknown", status)

    if is_stream:
        if status == TASK_STATUS_COMPLETED:
            push_to_session(task_id, SSEEvent.FINAL, {
                "answer": _tasks[task_id].get("results", {}).get("answer", ""),
                "status": "completed"
            })
        elif status == TASK_STATUS_FAILED:
            push_to_session(task_id, SSEEvent.ERROR, {
                "error": "任务执行失败",
                "status": "failed"
            })


def get_task_status(task_id: str) -> str:
    """获取任务状态"""
    _ensure_task(task_id)
    return _tasks[task_id]["status"]


def get_done_task_list(task_id: str) -> list:
    """获取已完成的节点列表"""
    _ensure_task(task_id)
    return _tasks[task_id]["done"]


def get_running_task_list(task_id: str) -> list:
    """获取正在运行的节点列表"""
    _ensure_task(task_id)
    return _tasks[task_id]["running"]


def set_task_result(session_id: str, key: str, value):
    """存储任务结果"""
    _ensure_task(session_id)
    _tasks[session_id]["results"][key] = value
    _persist_task_progress(session_id)


def get_task_result(session_id: str, key: str, default=None):
    """获取任务结果"""
    _ensure_task(session_id)
    return _tasks[session_id]["results"].get(key, default)
