"""简单的任务状态追踪工具（兼容旧接口）"""

from knowledge.utils.task_utils import (
    add_running_task as _add_running,
    add_done_task as _add_done,
    get_task_status as _get_status,
)
