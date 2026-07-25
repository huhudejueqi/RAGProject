"""缓存对象序列化。"""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from qa_core.retrieval.results import RetrievalHit, RetrievalResult


def retrieval_result_to_payload(result: RetrievalResult) -> dict[str, Any]:
    """将 RetrievalResult 转成 JSON 可存储结构。"""
    return {
        "query": result.query,
        "source_type": result.source_type,
        "elapsed_ms": result.elapsed_ms,
        "hits": [
            {
                "score": hit.score,
                "page_content": hit.document.page_content,
                "metadata": dict(hit.document.metadata),
            }
            for hit in result.hits
        ],
    }


def retrieval_result_from_payload(payload: dict[str, Any]) -> RetrievalResult:
    """从 JSON 结构恢复 RetrievalResult。"""
    hits = [
        RetrievalHit(
            document=Document(
                page_content=str(item.get("page_content") or ""),
                metadata=dict(item.get("metadata") or {}),
            ),
            score=float(item.get("score") or 0.0),
        )
        for item in payload.get("hits", [])
    ]
    return RetrievalResult(
        hits=hits,
        query=str(payload.get("query") or ""),
        source_type=payload.get("source_type") or "doc",
        elapsed_ms=float(payload.get("elapsed_ms") or 0.0),
    )
