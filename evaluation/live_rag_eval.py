"""在已配置的完整环境中运行端到端 RAG 质量门禁。"""

import argparse
import json
import uuid
from pathlib import Path

from evaluation.rag_evaluator import evaluate_cases, load_jsonl
from knowledge.processor.query_process.main_graph import KBQueryWorkflow


def run_live_cases(golden_cases: list[dict]) -> list[dict]:
    workflow = KBQueryWorkflow()
    results = []
    for case in golden_cases:
        case_id = case.get("id") or str(uuid.uuid4())
        state = workflow.run(
            {
                "task_id": f"eval-{case_id}",
                "session_id": f"eval-{case_id}",
                "original_query": case["query"],
                "answer": "",
                "is_stream": False,
            },
            thread_id=f"eval:{case_id}:{uuid.uuid4()}",
        )
        results.append({
            **case,
            "retrieved_ids": [
                document.get("chunk_id", "") for document in state.get("reranked_docs", [])
            ],
            "answer": state.get("answer", ""),
        })
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="UEJAVA 端到端 RAG 质量门禁")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--min-score", type=float, default=0.75)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = run_live_cases(load_jsonl(args.dataset))
    metrics = evaluate_cases(results, args.k)
    if args.output:
        args.output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in results) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    quality_metrics = ["recall_at_k", "mrr", "ndcg_at_k", "keyword_coverage"]
    return 0 if all(metrics[name] >= args.min_score for name in quality_metrics) else 1


if __name__ == "__main__":
    raise SystemExit(main())
