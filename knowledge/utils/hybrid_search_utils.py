"""Milvus 混合搜索（稠密+稀疏加权融合）"""

from pymilvus import AnnSearchRequest, WeightedRanker


def create_hybrid_search_requests(dense_vector, sparse_vector, limit=5, expr=None):
    """
    构建 BGE-M3 稠密+稀疏混合搜索请求

    Args:
        dense_vector: 稠密向量 [1024维浮点数]
        sparse_vector: 稀疏向量 {词ID: 权重}
        limit: 返回数量
        expr: Milvus 过滤表达式，如 "facility_name in ['金风GW121']"
    """
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param={"metric_type": "COSINE"},
        limit=limit,
        expr=expr
    )
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param={"metric_type": "IP"},
        limit=limit,
        expr=expr
    )
    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.8, 0.2),
                  limit=5, norm_score=True, output_fields=None):
    """
    执行混合向量搜索（稠密+稀疏加权融合）

    优先使用新版 MilvusClient API，失败降级到旧版 Collection API
    """
    if output_fields is None:
        output_fields = ["facility_name"]

    ranker = WeightedRanker(*ranker_weights)

    # 尝试新版 API
    if hasattr(client, 'hybrid_search'):
        return client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=ranker,
            limit=limit,
            output_fields=output_fields
        )

    # 降级到旧版 API
    from pymilvus import Collection, connections
    from knowledge.configs.milvus_config import milvus_config
    connections.connect(alias="default", uri=milvus_config.milvus_url)
    collection = Collection(collection_name)
    collection.load()
    return collection.hybrid_search(
        reqs=reqs,
        rerank=ranker,
        limit=limit,
        output_fields=output_fields
    )
