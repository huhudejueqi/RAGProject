"""缓存 namespace 与 epoch 管理。

版本发布或回滚时推进 epoch，旧 Redis key 自然失效，不需要扫描删除全量缓存。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from qa_core.governance.data_scope import DEFAULT_DATASET_ID, DEFAULT_TENANT_ID
from qa_core.memory.base import _MySqlStore, safe_sql_identifier


CACHE_NAMESPACE_TABLE = "cache_namespaces"


def bump_cache_epoch_for_scenario_with_conn(conn, scenario_id: str) -> int:
    """在外部事务中推进某个场景下所有 namespace 的 epoch。

    执行逻辑：
    1. UPDATE ... SET cache_epoch = cache_epoch + 1 对所有已有 namespace 自增 epoch
    2. 如果 rowcount == 0（该场景还没有任何 namespace），插入默认 namespace：
       - 初始 epoch = 2（因为正常从 1 开始，这里模拟了一次 UPDATE +1 的效果，
         保证首次 bump 后新缓存 key 中的 epoch 为 2，而不是 1）
    """
    table = safe_sql_identifier(CACHE_NAMESPACE_TABLE, label="Cache namespace table")
    result = conn.execute(
        text(
            f"""
            UPDATE {table}
            SET cache_epoch = cache_epoch + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE scenario_id = :scenario_id
            """
        ),
        {"scenario_id": scenario_id},
    )
    if int(result.rowcount or 0) == 0:
        # 该场景尚无 namespace → 创建默认 namespace，epoch 从 2 开始
        # （模拟正常 UPDATE +1：如果没有这一步，新插入的 epoch=1 会与旧缓存 key 冲突）
        conn.execute(
            text(
                f"""
                INSERT INTO {table}
                    (scenario_id, tenant_id, dataset_id, cache_epoch)
                VALUES
                    (:scenario_id, :tenant_id, :dataset_id, 2)
                """
            ),
            {
                "scenario_id": scenario_id,
                "tenant_id": DEFAULT_TENANT_ID,
                "dataset_id": DEFAULT_DATASET_ID,
            },
        )
        return 1
    return int(result.rowcount or 0)


class CacheNamespaceStore(_MySqlStore):
    """缓存 namespace 元数据存储。"""

    def __init__(self) -> None:
        super().__init__()
        self.table = safe_sql_identifier(CACHE_NAMESPACE_TABLE, label="Cache namespace table")

    def get_epoch(self, *, scenario_id: str, tenant_id: str, dataset_id: str) -> int:
        """读取 namespace epoch；不存在时创建初始记录。

        使用 INSERT ... ON DUPLICATE KEY UPDATE cache_epoch = cache_epoch 的 MySQL 技巧：
        - 如果记录不存在 → INSERT 创建 epoch=1 的初始记录
        - 如果记录已存在 → UPDATE 不改变 epoch 值（自赋值），配合 rowcount 判断是否新插入
        这个模式保证"不存在时自动创建"的语义，且不使用 SELECT-then-INSERT 的竞态方案。
        """
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    INSERT INTO {self.table}
                        (scenario_id, tenant_id, dataset_id, cache_epoch)
                    VALUES
                        (:scenario_id, :tenant_id, :dataset_id, 1)
                    ON DUPLICATE KEY UPDATE
                        cache_epoch = cache_epoch
                    """
                ),
                {"scenario_id": scenario_id, "tenant_id": tenant_id, "dataset_id": dataset_id},
            )
            row = conn.execute(
                text(
                    f"""
                    SELECT cache_epoch
                    FROM {self.table}
                    WHERE scenario_id=:scenario_id
                      AND tenant_id=:tenant_id
                      AND dataset_id=:dataset_id
                    """
                ),
                {"scenario_id": scenario_id, "tenant_id": tenant_id, "dataset_id": dataset_id},
            ).mappings().fetchone()
        return int(row["cache_epoch"] or 1)

    def bump_scenario_epoch(self, scenario_id: str) -> int:
        """推进场景下所有 namespace 的 epoch。"""
        with self.engine.begin() as conn:
            return bump_cache_epoch_for_scenario_with_conn(conn, scenario_id)

    def list_namespaces(self, *, scenario_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """返回 namespace 列表，用于管理页查看缓存视图。"""
        safe_limit = max(1, min(int(limit or 100), 500))
        where = "WHERE scenario_id=:scenario_id" if scenario_id else ""
        params = {"scenario_id": scenario_id} if scenario_id else {}
        sql = f"""
        SELECT scenario_id, tenant_id, dataset_id, cache_epoch, updated_at
        FROM {self.table}
        {where}
        ORDER BY updated_at DESC
        LIMIT {safe_limit}
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [dict(row) for row in rows]
