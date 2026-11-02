"""检索工具参数读取。"""


def tool_query(state: dict, name: str) -> str:
    """优先执行模型为该工具提供的查询参数。"""
    return (
        state.get("tool_queries", {}).get(name)
        or state.get("rewritten_query")
        or state.get("original_query", "")
    )
