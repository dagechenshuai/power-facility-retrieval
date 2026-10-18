from knowledge.processor.query_process.nodes.node_rerank import NodeRerank
from knowledge.processor.query_process.nodes.node_rrf import NodeRrf
from knowledge.processor.query_process.nodes.node_search_embedding import NodeSearchEmbedding
from knowledge.utils.bm25_store import replace_document, search_bm25


def test_parent_child_retrieval_deduplicates_and_returns_parent():
    hits = [
        {"entity": {"chunk_id": "1", "parent_id": "p1", "content": "子块一", "parent_content": "完整父块"}},
        {"entity": {"chunk_id": "2", "parent_id": "p1", "content": "子块二", "parent_content": "完整父块"}},
    ]
    parents = NodeSearchEmbedding._parents(hits)
    assert parents == [{
        "chunk_id": "parent:p1", "parent_id": "p1", "content": "完整父块",
        "parent_content": "完整父块", "source": "parent_vector",
    }]


def test_rrf_rewards_documents_found_by_multiple_retrievers():
    merged = NodeRrf()._rrf_merge([
        ([{"chunk_id": "shared"}, {"chunk_id": "vector-only"}], 1.0),
        ([{"chunk_id": "shared"}, {"chunk_id": "bm25-only"}], 1.0),
    ])
    assert merged[0][0]["chunk_id"] == "shared"


def test_rerank_cliff_cutoff():
    docs = [{"score": score} for score in [0.95, 0.9, 0.85, 0.2, 0.1]]
    assert len(NodeRerank()._step_3_cliff_cutoff(docs)) == 3


def test_traditional_bm25_supports_chinese_terms():
    replace_document("pytest-bm25", [{
        "parent_id": "bm25-parent",
        "file_title": "pytest-bm25",
        "facility_name": "风力发电机",
        "title": "维护手册",
        "content": "叶片检查",
        "parent_content": "风力发电机的叶片需要每半年检查一次。",
    }])
    results = search_bm25("风力发电机叶片检查", ["风力发电机"])
    assert results[0]["parent_id"] == "bm25-parent"
    assert "每半年" in results[0]["content"]
