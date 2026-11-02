"""MCP 联网搜索节点：调用百炼 WebSearch 获取最新网络信息"""

import asyncio
import json
import logging

from agents.mcp import MCPServerStreamableHttp

from knowledge.configs.mcp_config import mcp_config
from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.tool_arguments import tool_query


class NodeWebSearchMcp(NodeBase):
    """节点5（可选）：MCP 联网搜索 —— 获取电力设施相关的最新网络信息"""

    name = "node_web_search_mcp"

    def process(self, state: QueryGraphState):
        query = tool_query(state, "web_search")
        docs = []

        if query:
            try:
                result = asyncio.run(self._mcp_call(query))
                if result:
                    pages = json.loads(result.content[0].text).get("pages") or []
                    for item in pages:
                        snippet = (item.get("snippet") or "").strip()
                        url = (item.get("url") or "").strip()
                        title = (item.get("title") or "").strip()
                        if not snippet:
                            continue
                        docs.append({"title": title, "url": url, "snippet": snippet})
                    self.logger.info(f"MCP 搜索结果: {len(docs)} 条")
            except Exception as e:
                self.logger.error(f"MCP 搜索失败: {e}")

        state["web_search_docs"] = docs
        return state

    async def _mcp_call(self, query: str):
        """调用百炼 MCP WebSearch"""
        search_mcp = MCPServerStreamableHttp(
            name="web_search",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=2,
        )
        try:
            await search_mcp.connect()
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )
            return result
        finally:
            await search_mcp.cleanup()
