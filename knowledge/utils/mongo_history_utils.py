"""MongoDB 对话历史管理"""

import os
from datetime import datetime
from typing import List, Dict

from pymongo import MongoClient
from bson import ObjectId


class HistoryMongoTool:
    """MongoDB 对话历史管理工具"""

    def __init__(self):
        mongo_url = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017")
        mongo_db = os.getenv("MONGO_DB_NAME", "uejava_agent")
        self.client = MongoClient(mongo_url)
        self.db = self.client[mongo_db]
        self.collection = self.db["chat_message"]
        # 创建复合索引加速查询
        self.collection.create_index([("session_id", 1), ("ts", -1)])

    def clear_history(self, session_id: str) -> int:
        """清空指定会话的所有消息"""
        result = self.collection.delete_many({"session_id": session_id})
        return result.deleted_count

    def save_chat_message(
        self, session_id: str, role: str, text: str,
        rewritten_query: str = "", facility_names=None, image_urls=None,
        message_id: str = None
    ) -> str:
        """保存或更新聊天消息"""
        doc = {
            "session_id": session_id,
            "role": role,
            "text": text,
            "rewritten_query": rewritten_query,
            "facility_names": facility_names or [],
            "image_urls": image_urls or [],
            "ts": datetime.utcnow(),
        }
        if message_id:
            self.collection.update_one(
                {"_id": ObjectId(message_id)},
                {"$set": doc}
            )
            return message_id
        else:
            result = self.collection.insert_one(doc)
            return str(result.inserted_id)

    def update_message_facility_names(self, ids: List[str], facility_names: List[str]) -> int:
        """批量更新消息的设施名称"""
        obj_ids = [ObjectId(mid) for mid in ids]
        result = self.collection.update_many(
            {"_id": {"$in": obj_ids}},
            {"$set": {"facility_names": facility_names}}
        )
        return result.modified_count

    def get_recent_messages(self, session_id: str, limit: int = 10) -> List[Dict]:
        """获取最近的消息记录，按时间升序"""
        cursor = self.collection.find({"session_id": session_id}) \
            .sort("ts", -1).limit(limit)
        records = list(cursor)
        records.reverse()  # 升序返回
        return records


_history_tool = None


def get_history_mongo_tool() -> HistoryMongoTool:
    """获取 HistoryMongoTool 单例"""
    global _history_tool
    if _history_tool is None:
        _history_tool = HistoryMongoTool()
    return _history_tool


def clear_history(session_id: str) -> int:
    return get_history_mongo_tool().clear_history(session_id)


def save_chat_message(session_id: str, role: str, text: str,
                      rewritten_query: str = "", facility_names=None,
                      image_urls=None, message_id: str = None) -> str:
    return get_history_mongo_tool().save_chat_message(
        session_id, role, text,
        rewritten_query=rewritten_query,
        facility_names=facility_names,
        image_urls=image_urls,
        message_id=message_id
    )


def update_message_facility_names(ids: List[str], facility_names: List[str]) -> int:
    return get_history_mongo_tool().update_message_facility_names(ids, facility_names)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict]:
    return get_history_mongo_tool().get_recent_messages(session_id, limit)
