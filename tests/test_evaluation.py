from evaluation.rag_evaluator import evaluate_cases, ndcg_at_k, reciprocal_rank


def test_retrieval_and_answer_metrics():
    metrics = evaluate_cases([{
        "retrieved_ids": ["p2", "p1", "p3"],
        "relevant_ids": ["p1", "p2"],
        "answer": "风机建议每半年进行一次维护。",
        "answer_keywords": ["半年", "维护"],
    }], k=2)

    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["keyword_coverage"] == 1.0
    assert ndcg_at_k(["x", "p1"], ["p1"], 2) < 1.0
    assert reciprocal_rank(["x", "p1"], ["p1"]) == 0.5
