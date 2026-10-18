"""Spring Boot 后端 REST API 客户端

通过 HTTP 调用 UEJAVA 的 Spring Boot 服务获取模型、场景、地理数据的实时信息。
"""

import logging
from typing import Dict, List, Optional

import httpx
from knowledge.configs.springboot_config import springboot_config

logger = logging.getLogger("agent.springboot")


class SpringBootClient:
    """Spring Boot 后端异步 HTTP 客户端"""

    def __init__(self):
        self.base_url = springboot_config.base_url.rstrip("/")
        self.timeout = springboot_config.timeout_seconds
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def _get(self, path: str) -> dict:
        """发送 GET 请求到 Spring Boot"""
        url = f"{self.base_url}{path}"
        try:
            resp = self.client.get(url)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Spring Boot API 请求失败 [{url}]: {e}")
            return {}

    # ==================== 模型相关 ====================

    def get_all_models(self) -> dict:
        """获取所有模型列表"""
        return self._get("/api/models")

    def get_models_with_coordinates(self) -> dict:
        """获取带坐标的模型列表"""
        return self._get("/api/models/with-coordinates")

    def get_combined_models(self, project_id: str = None) -> dict:
        """获取合并模型列表（本地+外部同步）"""
        params = f"?projectId={project_id}" if project_id else ""
        return self._get(f"/api/combined-models{params}")

    def get_model_coordinates(self, model_id: str) -> dict:
        """获取指定模型的坐标信息"""
        return self._get(f"/api/models/{model_id}/coordinates")

    def get_model_detail(self, model_id: str) -> dict:
        """获取指定模型的完整数据库记录"""
        return self._get(f"/api/models/{model_id}/debug")

    # ==================== 场景相关 ====================

    def get_all_scenes(self) -> dict:
        """获取所有已发布场景列表"""
        return self._get("/api/published-scenes")

    def get_scene_detail(self, scene_id: str) -> dict:
        """获取指定场景详情"""
        return self._get(f"/api/published-scenes/{scene_id}")

    def get_scene_models(self, scene_id: str) -> list:
        """获取场景下的模型列表"""
        return self._get(f"/api/scenes/{scene_id}/models")

    def get_scene_resolved_models(self, scene_id: str) -> dict:
        """获取场景下已解析的模型（哈希→实名映射）"""
        return self._get(f"/api/scenes/{scene_id}/resolved-models")

    def get_scene_stats(self) -> dict:
        """获取场景统计信息"""
        return self._get("/api/published-scenes/stats")

    # ==================== 地理数据相关 ====================

    def get_geo_items(self, geo_type: str = None) -> list:
        """获取地理数据项列表"""
        params = f"?type={geo_type}" if geo_type else ""
        return self._get(f"/api/geo-items{params}")

    def get_terrain_items(self) -> list:
        """获取地形数据列表"""
        return self._get("/api/terrain-items")

    def get_imagery_items(self) -> list:
        """获取影像数据列表"""
        return self._get("/api/imagery-items")

    def get_geo_item_detail(self, geo_id: str) -> dict:
        """获取指定地理数据项详情"""
        return self._get(f"/api/geo-items/{geo_id}")

    # ==================== 搜索辅助 ====================

    def search_by_keyword(self, keyword: str) -> List[Dict]:
        """
        根据关键词在模型名、场景名中搜索相关记录

        Args:
            keyword: 搜索关键词（如"红光"、"变电站"、"风机"）

        Returns:
            [{"type": "model", "data": {...}}, {"type": "scene", "data": {...}}]
        """
        results = []

        # 搜索模型
        models = self.get_all_models()
        for item in models.get("items", []):
            name = item.get("name", "")
            if keyword and keyword.lower() in name.lower():
                results.append({"type": "model", "data": item})

        # 搜索场景
        scenes = self.get_all_scenes()
        scenes_list = scenes if isinstance(scenes, list) else scenes.get("items", []) if isinstance(scenes, dict) else []
        for item in scenes_list if isinstance(scenes_list, list) else []:
            name = item.get("sceneName", "")
            if keyword and keyword.lower() in name.lower():
                results.append({"type": "scene", "data": item})

        return results


# 全局单例
_springboot_client: Optional[SpringBootClient] = None


def get_springboot_client() -> SpringBootClient:
    """获取 SpringBootClient 单例"""
    global _springboot_client
    if _springboot_client is None:
        _springboot_client = SpringBootClient()
    return _springboot_client
