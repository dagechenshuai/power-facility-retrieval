"""模型/场景检索节点：调用 Spring Boot API 获取实时模型和场景数据"""

import logging
from typing import Dict, List

from knowledge.processor.query_process.base import NodeBase
from knowledge.processor.query_process.state import QueryGraphState
from knowledge.processor.query_process.tool_arguments import tool_query
from knowledge.utils.springboot_client import get_springboot_client


class NodeModelSceneSearch(NodeBase):
    """
    节点4：调用 Spring Boot 后端 API 获取模型、场景、地理数据的实时信息

    根据查询内容智能选择调用的 API：
    - 涉及"模型"、"风机"、"变电站" → 调用模型 API
    - 涉及"场景"、"发布" → 调用场景 API
    - 涉及"地形"、"影像"、"坐标" → 调用地理数据 API
    - 综合问题 → 调用多个 API
    """

    name = "node_model_scene_search"

    def process(self, state: QueryGraphState):
        query = tool_query(state, "scene_search")
        facility_names = state.get("facility_names", [])
        results = []

        client = get_springboot_client()

        # 根据查询内容决定调用哪些 API
        query_lower = query.lower()

        # 1. 搜索相关模型
        if self._should_search_models(query_lower):
            models = self._search_models(client, facility_names, query)
            results.extend(models)

        # 2. 搜索相关场景
        if self._should_search_scenes(query_lower):
            scenes = self._search_scenes(client, facility_names, query)
            results.extend(scenes)

        # 3. 搜索地理数据
        if self._should_search_geo(query_lower):
            geo_data = self._search_geo(client, facility_names, query)
            results.extend(geo_data)

        # 如果没有特定匹配，获取概览数据
        if not results:
            results = self._get_overview(client)

        self.logger.info(f"Spring Boot API 查询完成，返回 {len(results)} 条")
        state["backend_api_data"] = results
        return state

    def _should_search_models(self, query_lower: str) -> bool:
        keywords = [
            "模型",
            "model",
            "glb",
            "fbx",
            "风机",
            "变电站",
            "设备",
            "变压器",
            "3d",
            "三维",
        ]
        return any(kw in query_lower for kw in keywords)

    def _should_search_scenes(self, query_lower: str) -> bool:
        keywords = ["场景", "scene", "发布", "publish", "项目"]
        return any(kw in query_lower for kw in keywords)

    def _should_search_geo(self, query_lower: str) -> bool:
        keywords = [
            "地形",
            "影像",
            "坐标",
            "经纬度",
            "高度",
            "海拔",
            "地理",
            "dem",
            "terrain",
            "imagery",
            "geo",
            "位置",
        ]
        return any(kw in query_lower for kw in keywords)

    def _search_models(
        self, client, facility_names: List[str], query: str
    ) -> List[Dict]:
        """搜索模型数据"""
        results = []
        try:
            # 尝试获取带坐标的模型
            data = client.get_models_with_coordinates()
            items = data.get("items", [])
            self.logger.info(f"获取到 {len(items)} 个带坐标的模型")

            # 如果有设施名，过滤匹配的
            relevant_items = self._filter_by_keywords(
                items, facility_names, ["name", "type"], max_results=5
            )
            for item in relevant_items:
                coord_str = ""
                if item.get("longitude") and item.get("latitude"):
                    coord_str = f"（经度: {item['longitude']}, 纬度: {item['latitude']}"
                    if item.get("height"):
                        coord_str += f", 高度: {item['height']}m"
                    coord_str += "）"
                results.append(
                    {
                        "content": f"3D模型「{item.get('name', '')}」类型: {item.get('type', '')}{coord_str}",
                        "title": f"模型: {item.get('name', '')}",
                        "chunk_id": item.get("id", ""),
                        "source": "backend_model",
                    }
                )
        except Exception as e:
            self.logger.error(f"模型搜索失败: {e}")

        return results

    def _search_scenes(
        self, client, facility_names: List[str], query: str
    ) -> List[Dict]:
        """搜索场景数据"""
        results = []
        try:
            data = client.get_all_scenes()
            items = []
            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("items", data.get("data", []))

            if isinstance(items, list):
                self.logger.info(f"获取到 {len(items)} 个场景")
                relevant = self._filter_by_keywords(
                    items, facility_names, ["sceneName", "sceneId"], max_results=3
                )
                for item in relevant:
                    name = item.get("sceneName", item.get("sceneId", ""))
                    created = item.get("createdAt", "")
                    results.append(
                        {
                            "content": f"已发布场景「{name}」创建时间: {created}",
                            "title": f"场景: {name}",
                            "chunk_id": item.get("sceneId", ""),
                            "source": "backend_scene",
                        }
                    )
        except Exception as e:
            self.logger.error(f"场景搜索失败: {e}")

        return results

    def _search_geo(self, client, facility_names: List[str], query: str) -> List[Dict]:
        """搜索地理数据"""
        results = []
        try:
            geo_items = client.get_geo_items()
            items = []
            if isinstance(geo_items, list):
                items = geo_items
            elif isinstance(geo_items, dict):
                items = geo_items.get("items", [])

            self.logger.info(f"获取到 {len(items)} 个地理数据项")
            relevant = self._filter_by_keywords(
                items, facility_names, ["name", "description", "type"], max_results=3
            )
            for item in relevant:
                results.append(
                    {
                        "content": f"地理数据「{item.get('name', '')}」类型: {item.get('type', '')}, "
                        f"子类型: {item.get('subType', '')}, 坐标系: {item.get('crs', '')}",
                        "title": f"地理: {item.get('name', '')}",
                        "chunk_id": item.get("id", ""),
                        "source": "backend_geo",
                    }
                )
        except Exception as e:
            self.logger.error(f"地理数据搜索失败: {e}")

        return results

    def _get_overview(self, client) -> List[Dict]:
        """获取系统概览数据（兜底）"""
        results = []
        try:
            stats = client.get_scene_stats()
            if stats:
                results.append(
                    {
                        "content": f"系统概览：共 {stats.get('totalScenes', '?')} 个已发布场景，"
                        f"总计 {stats.get('totalZipSize', '?')} 字节",
                        "title": "系统概览",
                        "source": "backend_overview",
                    }
                )
        except Exception:
            pass
        return results

    def _filter_by_keywords(
        self,
        items: List[Dict],
        keywords: List[str],
        search_fields: List[str],
        max_results: int = 5,
    ) -> List[Dict]:
        """按关键词过滤和排序"""
        if not keywords:
            return items[:max_results]

        scored = []
        for item in items:
            score = 0
            for field in search_fields:
                value = str(item.get(field, "")).lower()
                for kw in keywords:
                    if kw.lower() in value:
                        score += 1
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:max_results]]
