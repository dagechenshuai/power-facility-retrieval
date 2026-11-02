"""用合成资料演示真实 BM25 入库和检索；不调用模型。"""

import json
import os
import tempfile
from pathlib import Path


def main() -> None:
    """使用独立临时数据库，避免污染已有知识库。"""
    database_dir = Path(tempfile.mkdtemp(prefix="power-bm25-demo-"))
    os.environ["BM25_DB_PATH"] = str(database_dir / "bm25.db")
    from knowledge.utils.bm25_store import replace_document, search_bm25

    documents = json.loads(
        Path("examples/demo-documents.json").read_text(encoding="utf-8-sig")
    )
    replace_document("synthetic-demo", documents)
    output = {
        "mode": "local_retrieval_only",
        "boundary": "合成资料，仅演示BM25和父块返回，不证明真实模型决策或企业检索指标。",
        "query": "风机叶片检查记录",
        "results": search_bm25("风机叶片检查记录"),
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
