"""BGE-M3 混合向量嵌入工具。"""

from knowledge.configs.embedding_config import embedding_config

# BGE-M3 模型单例
_bge_m3_ef = None


def get_bge_m3_ef():
    """获取 BGE-M3 模型单例，第一次调用时加载模型（耗时约几秒）"""
    global _bge_m3_ef
    if _bge_m3_ef is not None:
        return _bge_m3_ef

    from FlagEmbedding import BGEM3FlagModel

    model_name = embedding_config.bge_m3_path or embedding_config.bge_m3
    device = embedding_config.bge_device
    use_fp16 = embedding_config.bge_fp16

    _bge_m3_ef = BGEM3FlagModel(
        model_name,
        devices=device,
        use_fp16=use_fp16
    )
    return _bge_m3_ef


def generate_embeddings(texts: list) -> dict:
    """
    将文本列表转为稠密+稀疏向量

    Args:
        texts: 文本列表，如 ["金风GW121风机说明书", "松滋红光变电站"]

    Returns:
        {
            "dense": [[1024维浮点数...], ...],
            "sparse": [{词ID: 权重}, ...]
        }
    """
    model = get_bge_m3_ef()
    embeddings = model.encode(
        texts,
        batch_size=embedding_config.embedding_batch_size,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    processed_sparse = [
        {int(token_id): float(weight) for token_id, weight in weights.items()}
        for weights in embeddings["lexical_weights"]
    ]

    return {
        "dense": [embedding.tolist() for embedding in embeddings["dense_vecs"]],
        "sparse": processed_sparse
    }
