"""缓存存储适配器。"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from typing import Any

from qa_core.config.settings import get_settings


@dataclass
class _MemoryItem:
    value: Any
    expires_at: float


class TTLMemoryCache:
    """进程内短 TTL 缓存，用于 cache namespace epoch 等低频元数据。"""

    def __init__(self) -> None:
        self._items: dict[str, _MemoryItem] = {}

    def get(self, key: str) -> Any | None:
        """读取缓存键；未命中或过期时返回空值。

        调用顺序：测试或业务入口 -> TTLMemoryCache.get()。
        """
        item = self._items.get(key)
        if item is None:
            return None
        if item.expires_at <= time.time():
            self._items.pop(key, None)
            return None
        return item.value

    def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        """写入缓存值并应用调用方提供的过期时间。

        调用顺序：测试或业务入口 -> TTLMemoryCache.set()。
        """
        self._items[key] = _MemoryItem(value=value, expires_at=time.time() + max(1, ttl_seconds))

    def clear(self) -> None:
        """清空当前进程缓存，供版本切换和测试隔离使用。

        调用顺序：测试或业务入口 -> TTLMemoryCache.clear()。
        """
        self._items.clear()


class RedisJsonCache:
    """Redis JSON 缓存适配器。"""

    def __init__(self, client: Any | None = None) -> None:
        settings = get_settings()
        if client is not None:
            self.client = client
            return
        try:
            redis_module = importlib.import_module("redis")
        except ImportError as exc:
            raise RuntimeError("Redis 缓存已启用，但未安装 redis 依赖。请安装 requirements.txt。") from exc
        self.client = redis_module.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_socket_timeout,
        )

    def get_json(self, key: str) -> dict[str, Any] | None:
        """读取并反序列化 JSON 缓存值。

        调用顺序：测试或业务入口 -> RedisJsonCache.get_json()。
        """
        raw = self.client.get(key)
        if not raw:
            return None
        return json.loads(raw)

    def set_json(self, key: str, payload: dict[str, Any], ttl_seconds: int) -> None:
        """序列化对象并写入带过期时间的 JSON 缓存。

        调用顺序：测试或业务入口 -> RedisJsonCache.set_json()。
        """
        self.client.set(key, json.dumps(payload, ensure_ascii=False), ex=max(1, int(ttl_seconds)))

    def delete_pattern(self, pattern: str) -> int:
        """按 pattern 批量删除 Redis key。

        使用 scan_iter 而非 keys()：scan_iter 分批迭代（count=500），
        不会阻塞 Redis 主线程。keys() 在 key 数量大时可能导致 Redis 短暂不可用。
        """
        keys = list(self.client.scan_iter(match=pattern, count=500))
        if not keys:
            return 0
        return int(self.client.delete(*keys))

    def ping(self) -> bool:
        """检查缓存后端连接是否可用。

        调用顺序：测试或业务入口 -> RedisJsonCache.ping()。
        """
        return bool(self.client.ping())
