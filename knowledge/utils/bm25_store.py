"""基于 SQLite FTS5 BM25 的可持久化传统关键词检索。"""

import os
import sqlite3
import threading
from pathlib import Path

import jieba

_PATH = Path(os.getenv("BM25_DB_PATH", "./data/bm25.db")).resolve()
_PATH.parent.mkdir(parents=True, exist_ok=True)
_CONN = sqlite3.connect(_PATH, check_same_thread=False)
_LOCK = threading.RLock()
_CONN.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS bm25_chunks_v2 USING fts5(
    parent_id UNINDEXED, file_title UNINDEXED, facility_name UNINDEXED,
    title_raw UNINDEXED, content_raw UNINDEXED, title_tokens, content_tokens,
    tokenize='unicode61'
)""")


def _tokenize(text: str) -> str:
    """使用结巴切词后交给 FTS5，避免中文整句被视作单一词项。"""
    return " ".join(
        token.strip().lower() for token in jieba.lcut(text or "") if token.strip()
    )


def replace_document(file_title: str, chunks: list[dict]) -> None:
    parents: dict[str, dict] = {}
    for item in chunks:
        parent_id = item.get("parent_id", "")
        parents.setdefault(parent_id, item)
    with _LOCK:
        _CONN.execute("DELETE FROM bm25_chunks_v2 WHERE file_title=?", (file_title,))
        _CONN.executemany(
            """INSERT INTO bm25_chunks_v2(
                parent_id,file_title,facility_name,title_raw,content_raw,title_tokens,content_tokens
            ) VALUES(?,?,?,?,?,?,?)""",
            [
                (
                    item.get("parent_id", ""),
                    file_title,
                    item.get("facility_name", ""),
                    item.get("title", ""),
                    item.get("parent_content") or item.get("content", ""),
                    _tokenize(item.get("title", "")),
                    _tokenize(item.get("parent_content") or item.get("content", "")),
                )
                for item in parents.values()
            ],
        )
        _CONN.commit()


def search_bm25(
    query: str, facility_names: list[str] | None = None, limit: int = 10
) -> list[dict]:
    terms = list(
        dict.fromkeys(term.replace('"', "") for term in _tokenize(query).split())
    )
    if not terms:
        return []
    match = " OR ".join(f'"{term}"' for term in terms)
    sql = """SELECT parent_id,file_title,facility_name,title_raw,content_raw,
             bm25(bm25_chunks_v2, 0, 0, 0, 0, 0, 5.0, 1.0) score
             FROM bm25_chunks_v2 WHERE bm25_chunks_v2 MATCH ?"""
    params: list = [match]
    if facility_names:
        sql += " AND facility_name IN (%s)" % ",".join("?" for _ in facility_names)
        params.extend(facility_names)
    sql += " ORDER BY score LIMIT ?"
    params.append(limit)
    with _LOCK:
        rows = _CONN.execute(sql, params).fetchall()
    return [
        {
            "chunk_id": f"parent:{row[0]}",
            "parent_id": row[0],
            "file_title": row[1],
            "facility_name": row[2],
            "title": row[3],
            "content": row[4],
            "source": "bm25",
            "score": float(-row[5]),
        }
        for row in rows
    ]
