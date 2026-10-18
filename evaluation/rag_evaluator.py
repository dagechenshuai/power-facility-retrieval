"""离线 RAG 回归评测：Recall@K、MRR、nDCG 与答案关键词覆盖率。"""

import argparse
import json
import math
from pathlib import Path


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    expected = set(relevant)
    if not expected:
        return 1.0
    return len(expected.intersection(retrieved[:k])) / len(expected)


def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    expected = set(relevant)
    for rank, document_id in enumerate(retrieved, start=1):
        if document_id in expected:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    expected = set(relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, document_id in enumerate(retrieved[:k], start=1)
        if document_id in expected
    )
    ideal_hits = min(len(expected), k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / ideal if ideal else 1.0


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 1.0
    normalized = answer.lower()
    return sum(keyword.lower() in normalized for keyword in keywords) / len(keywords)


def evaluate_cases(cases: list[dict], k: int = 5) -> dict[str, float]:
    if not cases:
        raise ValueError("评测集不能为空")
    rows = []
    for case in cases:
        retrieved = case.get("retrieved_ids", [])
        relevant = case.get("relevant_ids", [])
        rows.append({
            "recall_at_k": recall_at_k(retrieved, relevant, k),
            "mrr": reciprocal_rank(retrieved, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved, relevant, k),
            "keyword_coverage": keyword_coverage(
                case.get("answer", ""), case.get("answer_keywords", [])
            ),
        })
    return {
        metric: sum(row[metric] for row in rows) / len(rows)
        for metric in rows[0]
    } | {"case_count": len(rows)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description="UEJAVA RAG 离线质量门禁")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-recall", type=float, default=0.8)
    parser.add_argument("--min-mrr", type=float, default=0.7)
    parser.add_argument("--min-ndcg", type=float, default=0.7)
    parser.add_argument("--min-keyword-coverage", type=float, default=0.8)
    args = parser.parse_args()
    metrics = evaluate_cases(load_jsonl(args.dataset), args.k)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    passed = (
        metrics["recall_at_k"] >= args.min_recall
        and metrics["mrr"] >= args.min_mrr
        and metrics["ndcg_at_k"] >= args.min_ndcg
        and metrics["keyword_coverage"] >= args.min_keyword_coverage
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
