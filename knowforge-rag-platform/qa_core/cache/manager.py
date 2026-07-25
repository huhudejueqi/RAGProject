"""统一缓存入口。"""

from __future__ import annotations

import time
from functools import lru_cache

from qa_core.cache.keys import key_digest, retrieval_cache_key
from qa_core.cache.namespaces import CacheNamespaceStore
from qa_core.cache.serialization import retrieval_result_from_payload, retrieval_result_to_payload
from qa_core.cache.stores import RedisJsonCache, TTLMemoryCache
from qa_core.config.settings import get_settings
from qa_core.governance.data_scope import DataScope
from qa_core.retrieval.results import RetrievalResult


class CacheManager:
    """封装 L1 进程缓存、L2 Redis 缓存和 MySQL namespace epoch。"""

    def __init__(
        self,
        *,
        redis_cache: RedisJsonCache | None = None,
        namespace_store: CacheNamespaceStore | None = None,
        l1_cache: TTLMemoryCache | None = None,
        enabled: bool | None = None,
    ) -> None:
        self.settings = get_settings()
        self.enabled = self.settings.cache_enabled if enabled is None else enabled
        self.redis_cache = redis_cache
        self.namespace_store = namespace_store
        self.l1_cache = l1_cache or TTLMemoryCache()
        self._stats: dict[str, int] = {
            "retrieval_hits": 0,
            "retrieval_misses": 0,
            "retrieval_writes": 0,
            "embedding_hits": 0,
            "embedding_misses": 0,
            "embedding_writes": 0,
            "invalidations": 0,
        }
        self._recent_events: list[dict[str, object]] = []

    def _redis(self) -> RedisJsonCache:
        if self.redis_cache is None:
            self.redis_cache = RedisJsonCache()
        return self.redis_cache

    def _namespace_store(self) -> CacheNamespaceStore:
        if self.namespace_store is None:
            self.namespace_store = CacheNamespaceStore()
        return self.namespace_store

    def _record_event(self, *, kind: str, hit: bool | None = None, source_type: str = "", key: str | None = None) -> None:
        """记录进程级缓存观测事件。"""
        if kind in self._stats:
            self._stats[kind] += 1
        event = {
            "kind": kind,
            "hit": hit,
            "source_type": source_type,
            "key_digest": key.rsplit(":", 1)[-1] if key else "",
            "created_at": round(time.time(), 3),
        }
        self._recent_events.append(event)
        if len(self._recent_events) > 20:
            self._recent_events = self._recent_events[-20:]

    def namespace_epoch(self, *, scenario_id: str, data_scope: DataScope) -> int:
        """读取当前请求所属 namespace 的 epoch。"""
        scope = data_scope.as_dict()
        l1_key = f"epoch:{scenario_id}:{scope['tenant_id']}:{scope['dataset_id']}"
        cached = self.l1_cache.get(l1_key)
        if cached is not None:
            return int(cached)
        epoch = self._namespace_store().get_epoch(
            scenario_id=scenario_id,
            tenant_id=scope["tenant_id"],
            dataset_id=scope["dataset_id"],
        )
        self.l1_cache.set(l1_key, epoch, self.settings.cache_namespace_l1_ttl_seconds)
        return epoch

    def retrieval_key(
        self,
        *,
        kind: str,
        scenario_id: str,
        collection_name: str,
        source_type: str,
        data_scope: DataScope,
        kb_version: str,
        source_filter: str | None,
        query_variants: list[str],
        k: int,
        rerank: bool,
    ) -> str | None:
        """生成检索缓存 key；缓存关闭时返回 None。"""
        if not self.enabled:
            return None
        epoch = self.namespace_epoch(scenario_id=scenario_id, data_scope=data_scope)
        return retrieval_cache_key(
            kind=kind,
            scenario_id=scenario_id,
            collection_name=collection_name,
            source_type=source_type,
            data_scope=data_scope,
            kb_version=kb_version,
            cache_epoch=epoch,
            source_filter=source_filter,
            query_variants=query_variants,
            k=k,
            rerank=rerank,
        )

    def get_retrieval_result(self, key: str | None, *, source_type: str = "") -> RetrievalResult | None:
        """读取检索结果缓存。"""
        if not key or not self.enabled:
            return None
        payload = self._redis().get_json(key)
        hit = bool(payload)
        self._record_event(
            kind="retrieval_hits" if hit else "retrieval_misses",
            hit=hit,
            source_type=source_type,
            key=key,
        )
        return retrieval_result_from_payload(payload) if payload else None

    def set_retrieval_result(self, key: str | None, result: RetrievalResult, *, source_type: str) -> None:
        """写入检索结果缓存。"""
        if not key or not self.enabled:
            return
        ttl = self.settings.cache_faq_ttl_seconds if source_type == "faq" else self.settings.cache_doc_ttl_seconds
        self._redis().set_json(key, retrieval_result_to_payload(result), ttl)
        self._record_event(kind="retrieval_writes", source_type=source_type, key=key)

    def embedding_key(self, text: str) -> str | None:
        """生成 query embedding 缓存 key。"""
        if not self.enabled:
            return None
        digest = key_digest(
            {
                "kind": "embedding",
                "text": text,
                "embedding_model_version": self.settings.embedding_model_version,
                "embedding_model_path": self.settings.embedding_model_path,
            }
        )
        return f"{self.settings.cache_key_prefix}:embedding:{self.settings.embedding_model_version}:{digest}"

    def get_embedding_vector(self, key: str | None) -> list[float] | None:
        """读取 query embedding 缓存。"""
        if not key or not self.enabled:
            return None
        payload = self._redis().get_json(key)
        if not payload or not isinstance(payload.get("vector"), list):
            self._record_event(kind="embedding_misses", hit=False, source_type="embedding", key=key)
            return None
        self._record_event(kind="embedding_hits", hit=True, source_type="embedding", key=key)
        return [float(item) for item in payload["vector"]]

    def set_embedding_vector(self, key: str | None, vector: list[float]) -> None:
        """写入 query embedding 缓存。"""
        if not key or not self.enabled:
            return
        self._redis().set_json(key, {"vector": vector}, self.settings.cache_embedding_ttl_seconds)
        self._record_event(kind="embedding_writes", source_type="embedding", key=key)

    def invalidate_scenario(self, scenario_id: str) -> int:
        """推进场景 cache epoch，使旧缓存全部失效。"""
        if not self.enabled:
            return 0
        self.l1_cache.clear()
        affected = self._namespace_store().bump_scenario_epoch(scenario_id)
        self._record_event(kind="invalidations", source_type="namespace", key=scenario_id)
        return affected

    def status(self, *, scenario_id: str | None = None) -> dict[str, object]:
        """返回缓存运行状态。"""
        namespaces = self._namespace_store().list_namespaces(scenario_id=scenario_id) if self.enabled else []
        redis_ok = self._redis().ping() if self.enabled else False
        return {
            "enabled": self.enabled,
            "redis": {
                "ok": redis_ok,
                "host": self.settings.redis_host,
                "port": self.settings.redis_port,
                "db": self.settings.redis_db,
            },
            "config": {
                "key_prefix": self.settings.cache_key_prefix,
                "faq_ttl_seconds": self.settings.cache_faq_ttl_seconds,
                "doc_ttl_seconds": self.settings.cache_doc_ttl_seconds,
                "embedding_ttl_seconds": self.settings.cache_embedding_ttl_seconds,
                "namespace_l1_ttl_seconds": self.settings.cache_namespace_l1_ttl_seconds,
                "embedding_model_version": self.settings.embedding_model_version,
                "reranker_model_version": self.settings.reranker_model_version,
                "chunk_schema_version": self.settings.chunk_schema_version,
            },
            "stats": dict(self._stats),
            "recent_events": list(reversed(self._recent_events)),
            "namespaces": namespaces,
        }


@lru_cache(maxsize=1)
def get_cache_manager() -> CacheManager:
    """返回进程级 CacheManager。"""
    return CacheManager()
