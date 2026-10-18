"""UEJAVA 智能体 API 连通性测试

测试内容：
1. LLM API (DashScope Qwen) 连通性
2. Spring Boot 后端 API 连通性
3. Milvus 连通性
4. MongoDB 连通性
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).absolute().parent))
import requests
from dotenv import load_dotenv

load_dotenv(override=True)


def test_llm_api():
    """测试 LLM API 连通性"""
    print("\n" + "=" * 50)
    print("1. 测试 LLM API (DashScope Qwen)")
    print("=" * 50)

    base_url = os.getenv("OPENAI_API_BASE", "")
    api_key = os.getenv("OPENAI_API_KEY", "")

    if not api_key:
        print("❌ OPENAI_API_KEY 未配置")
        return False

    try:
        resp = requests.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "qwen-flash",
                "messages": [{"role": "user", "content": "说你好"}],
                "max_tokens": 10,
            },
            timeout=15,
        )
        if resp.status_code == 200:
            print(f"✅ LLM API 正常: {resp.json().get('choices', [{}])[0].get('message', {}).get('content', '')}")
            return True
        else:
            print(f"❌ LLM API 返回: {resp.status_code} {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"❌ LLM API 连接失败: {e}")
        return False


def test_springboot_api():
    """测试 Spring Boot 后端连通性"""
    print("\n" + "=" * 50)
    print("2. 测试 Spring Boot 后端 API")
    print("=" * 50)

    base_url = os.getenv("SPRINGBOOT_BASE_URL", "http://127.0.0.1:8081")

    # 测试模型接口
    try:
        resp = requests.get(f"{base_url}/api/models/with-coordinates", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            count = data.get("count", 0)
            print(f"✅ 模型接口正常: {count} 个带坐标的模型")
        else:
            print(f"⚠️ 模型接口返回: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ 模型接口不可达: {e}")

    # 测试场景接口
    try:
        resp = requests.get(f"{base_url}/api/published-scenes", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                print(f"✅ 场景接口正常: {len(data)} 个已发布场景")
            else:
                print(f"✅ 场景接口正常")
        else:
            print(f"⚠️ 场景接口返回: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ 场景接口不可达: {e}")

    # 测试地理数据接口
    try:
        resp = requests.get(f"{base_url}/api/geo-items", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, list):
                print(f"✅ 地理数据接口正常: {len(data)} 个数据项")
            else:
                print(f"✅ 地理数据接口正常")
        else:
            print(f"⚠️ 地理数据接口返回: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ 地理数据接口不可达: {e}")


def test_milvus():
    """测试 Milvus 连通性"""
    print("\n" + "=" * 50)
    print("3. 测试 Milvus 向量数据库")
    print("=" * 50)

    try:
        from pymilvus import MilvusClient
        milvus_url = os.getenv("MILVUS_URL", "http://127.0.0.1:19530")
        client = MilvusClient(uri=milvus_url)
        collections = client.list_collections()
        print(f"✅ Milvus 连接成功: {len(collections)} 个集合 {collections}")
    except Exception as e:
        print(f"⚠️ Milvus 连接失败: {e}")


def test_mongodb():
    """测试 MongoDB 连通性"""
    print("\n" + "=" * 50)
    print("4. 测试 MongoDB 文档数据库")
    print("=" * 50)

    try:
        from pymongo import MongoClient
        mongo_url = os.getenv("MONGO_URL", "mongodb://127.0.0.1:27017")
        mongo_db = os.getenv("MONGO_DB_NAME", "uejava_agent")
        client = MongoClient(mongo_url, serverSelectionTimeoutMS=5000)
        client.server_info()
        db = client[mongo_db]
        collections = db.list_collection_names()
        print(f"✅ MongoDB 连接成功: 数据库={mongo_db}, 集合={collections}")
    except Exception as e:
        print(f"⚠️ MongoDB 连接失败: {e}")


def test_bge_m3():
    """测试 BGE-M3 模型加载"""
    print("\n" + "=" * 50)
    print("5. 测试 BGE-M3 嵌入模型")
    print("=" * 50)

    try:
        from knowledge.utils.embedding_utils import generate_embeddings
        texts = ["金风科技 GW121-2500 风力发电机技术手册"]
        result = generate_embeddings(texts)
        dense_dim = len(result["dense"][0])
        sparse_count = len(result["sparse"][0])
        print(f"✅ BGE-M3 正常: 稠密维度={dense_dim}, 稀疏词数={sparse_count}")
    except Exception as e:
        print(f"⚠️ BGE-M3 加载失败: {e}")


if __name__ == "__main__":
    print("=" * 50)
    print("  UEJAVA 智能体 - 系统连通性测试")
    print("=" * 50)

    results = []

    results.append(("LLM API", test_llm_api()))
    test_springboot_api()
    test_milvus()
    test_mongodb()
    test_bge_m3()

    print("\n" + "=" * 50)
    print("  测试完成")
    print("=" * 50)
