"""Embedding 查询缓存包装器。"""

from __future__ import annotations

from typing import Any

from qa_core.cache.manager import get_cache_manager


class CachedEmbeddings:
    """只缓存查询向量，不缓存入库文档向量。"""

    def __init__(self, base_embeddings: Any) -> None:
        self.base_embeddings = base_embeddings

    def embed_query(self, text: str) -> list[float]:
        """缓存用户 query embedding，降低热点问题向量化成本。"""
        manager = get_cache_manager()
        key = manager.embedding_key(text)
        cached = manager.get_embedding_vector(key)
        if cached is not None:
            return cached
        vector = self.base_embeddings.embed_query(text)
        manager.set_embedding_vector(key, vector)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """文档入库向量不写 Redis，避免大批 chunk 污染业务缓存。"""
        return self.base_embeddings.embed_documents(texts)

    def __getattr__(self, name: str):
        return getattr(self.base_embeddings, name)
