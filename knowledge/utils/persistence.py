"""任务、事件与 LangGraph 检查点的持久化基础设施。"""

import json
import os
import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

_LOCK = threading.RLock()
_DB_PATH = Path(os.getenv("AGENT_STATE_DB", "./data/agent_state.db")).resolve()
_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
_CONN = sqlite3.connect(_DB_PATH, check_same_thread=False)
_CONN.row_factory = sqlite3.Row
_CONN.executescript(
    """
    PRAGMA journal_mode=WAL;
    PRAGMA busy_timeout=5000;
    CREATE TABLE IF NOT EXISTS agent_tasks (
        task_id TEXT PRIMARY KEY,
        task_type TEXT NOT NULL,
        status TEXT NOT NULL,
        payload TEXT NOT NULL DEFAULT '{}',
        result TEXT NOT NULL DEFAULT '{}',
        error TEXT,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS agent_events (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id TEXT NOT NULL,
        event TEXT NOT NULL,
        data TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS ix_agent_events_task_sequence
        ON agent_events(task_id, sequence);
    """
)
_CONN.commit()


def create_checkpointer() -> SqliteSaver:
    """为每个工作流创建独立连接，避免并发任务共享 SQLite 事务。"""
    connection = sqlite3.connect(_DB_PATH, check_same_thread=False)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=5000")
    return SqliteSaver(connection)


# 兼容直接使用检查点器的测试与旧代码；业务工作流使用独立实例。
CHECKPOINTER = create_checkpointer()


def upsert_task(task_id: str, task_type: str, status: str, payload: dict | None = None) -> None:
    with _LOCK:
        _CONN.execute(
            """INSERT INTO agent_tasks(task_id, task_type, status, payload)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET status=excluded.status,
                 payload=excluded.payload, updated_at=CURRENT_TIMESTAMP""",
            (task_id, task_type, status, json.dumps(payload or {}, ensure_ascii=False)),
        )
        _CONN.commit()


def update_task(task_id: str, *, status: str | None = None,
                result: dict | None = None, error: str | None = None) -> None:
    fields, values = [], []
    for name, value in (("status", status), ("result", result), ("error", error)):
        if value is not None:
            fields.append(f"{name}=?")
            values.append(json.dumps(value, ensure_ascii=False) if name == "result" else value)
    if not fields:
        return
    with _LOCK:
        values.append(task_id)
        _CONN.execute(
            f"UPDATE agent_tasks SET {', '.join(fields)}, updated_at=CURRENT_TIMESTAMP WHERE task_id=?",
            values,
        )
        _CONN.commit()


def get_task(task_id: str) -> dict | None:
    with _LOCK:
        row = _CONN.execute("SELECT * FROM agent_tasks WHERE task_id=?", (task_id,)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["payload"] = json.loads(item["payload"] or "{}")
    item["result"] = json.loads(item["result"] or "{}")
    return item


def append_event(task_id: str, event: str, data: dict) -> int:
    with _LOCK:
        cursor = _CONN.execute(
            "INSERT INTO agent_events(task_id, event, data) VALUES (?, ?, ?)",
            (task_id, event, json.dumps(data, ensure_ascii=False)),
        )
        _CONN.commit()
        return int(cursor.lastrowid)


def list_events(task_id: str, after_sequence: int = 0) -> list[dict]:
    with _LOCK:
        rows = _CONN.execute(
            "SELECT sequence, event, data FROM agent_events WHERE task_id=? AND sequence>? ORDER BY sequence",
            (task_id, after_sequence),
        ).fetchall()
    return [
        {"sequence": row["sequence"], "event": row["event"], "data": json.loads(row["data"])}
        for row in rows
    ]


def mark_interrupted_tasks(task_type: str) -> int:
    """服务重启时将未完成任务标记为可恢复的中断状态。"""
    with _LOCK:
        cursor = _CONN.execute(
            """UPDATE agent_tasks SET status='interrupted', updated_at=CURRENT_TIMESTAMP
               WHERE task_type=? AND status='processing'""",
            (task_type,),
        )
        _CONN.commit()
        return cursor.rowcount


def list_tasks(task_type: str, status: str | None = None) -> list[dict]:
    """按类型和状态列出持久任务，供启动恢复使用。"""
    sql = "SELECT task_id FROM agent_tasks WHERE task_type=?"
    params: list[str] = [task_type]
    if status:
        sql += " AND status=?"
        params.append(status)
    with _LOCK:
        task_ids = [row[0] for row in _CONN.execute(sql, params).fetchall()]
    return [task for task_id in task_ids if (task := get_task(task_id))]


def check_state_store() -> bool:
    """供 readiness 探针检查本地状态库。"""
    with _LOCK:
        return _CONN.execute("SELECT 1").fetchone()[0] == 1
