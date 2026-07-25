"""MySQL control-plane index for document chunk validity windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import text

from qa_core.memory.base import _MySqlStore, safe_sql_identifier


KB_CHUNK_VERSIONS_TABLE = "kb_chunk_versions"


@dataclass(frozen=True)
class ChunkVersionRecord:
    """A document chunk validity record mirrored from Milvus metadata.

    调用顺序：治理或版本管理入口 -> ChunkVersionRecord。
    """

    scenario_id: str
    chunk_id: str
    source: str
    kb_version: str
    file_path: str
    valid_from_seq: int
    valid_to_seq: int = 0

    @classmethod
    def from_row(cls, row: Any) -> "ChunkVersionRecord":
        """Build a record from a SQLAlchemy row mapping.

        调用顺序：治理或版本管理入口 -> ChunkVersionRecord.from_row()。
        """

        payload = dict(row)
        payload["valid_from_seq"] = int(payload.get("valid_from_seq") or 0)
        payload["valid_to_seq"] = int(payload.get("valid_to_seq") or 0)
        payload["file_path"] = str(payload.get("file_path") or "")
        return cls(**payload)


class ChunkVersionIndex(_MySqlStore):
    """Store chunk validity windows for governance and rebuild diagnostics.

    调用顺序：治理或版本管理入口 -> ChunkVersionIndex。
    """

    def __init__(self) -> None:
        """Initialize the chunk validity index. Schema is prepared by bootstrap.

        调用顺序：治理或版本管理入口 -> ChunkVersionIndex.__init__()。
        """

        super().__init__()
        self.table_name = safe_sql_identifier(KB_CHUNK_VERSIONS_TABLE, label="KB chunk versions table")

    def upsert_chunks(
        self,
        chunk_ids: list[str],
        *,
        scenario_id: str,
        source: str,
        kb_version: str,
        valid_from_seq: int,
        valid_to_seq: int = 0,
        file_path: str = "",
    ) -> None:
        """Insert or refresh chunk validity metadata.

        调用顺序：治理或版本管理入口 -> ChunkVersionIndex.upsert_chunks()。
        """

        if not chunk_ids:
            # 无 chunk id 时直接跳过，避免空列表执行无效的 SQL INSERT
            return
        if valid_from_seq <= 0:
            raise ValueError("valid_from_seq must be a positive version sequence")
        sql = f"""
        INSERT INTO {self.table_name}
            (
                scenario_id, chunk_id, source, kb_version, file_path,
                valid_from_seq, valid_to_seq
            )
        VALUES
            (
                :scenario_id, :chunk_id, :source, :kb_version, :file_path,
                :valid_from_seq, :valid_to_seq
            )
        ON DUPLICATE KEY UPDATE
            source=VALUES(source),
            kb_version=VALUES(kb_version),
            file_path=VALUES(file_path),
            valid_from_seq=VALUES(valid_from_seq),
            valid_to_seq=VALUES(valid_to_seq),
            updated_at=CURRENT_TIMESTAMP
        """
        rows = [
            {
                "scenario_id": scenario_id,
                "chunk_id": chunk_id,
                "source": source,
                "kb_version": kb_version,
                "file_path": file_path,
                "valid_from_seq": int(valid_from_seq),
                "valid_to_seq": int(valid_to_seq or 0),
            }
            for chunk_id in chunk_ids
        ]
        with self.engine.begin() as conn:
            conn.execute(text(sql), rows)

    def ensure_chunks_validity(
        self,
        chunk_ids: list[str],
        *,
        scenario_id: str,
        source: str,
        kb_version: str,
        valid_from_seq: int,
        file_path: str = "",
    ) -> None:
        """补齐引用基线版本 chunk 的有效期元数据。

        调用顺序：治理或版本管理入口 -> ChunkVersionIndex.ensure_chunks_validity()。
        """

        self.upsert_chunks(
            chunk_ids,
            scenario_id=scenario_id,
            source=source,
            kb_version=kb_version,
            valid_from_seq=valid_from_seq,
            valid_to_seq=0,
            file_path=file_path,
        )

    def expire_chunks(
        self,
        chunk_ids: list[str],
        *,
        scenario_id: str,
        source: str,
        kb_version: str,
        valid_from_seq: int,
        valid_to_seq: int,
        file_path: str = "",
    ) -> None:
        """Mark chunks as invisible from ``valid_to_seq`` onward.

        调用顺序：治理或版本管理入口 -> ChunkVersionIndex.expire_chunks()。
        """

        if not chunk_ids:
            return
        if valid_to_seq <= 0:
            raise ValueError("valid_to_seq must be a positive version sequence")
        # 先确保有效期记录存在，再更新关闭序列号。
        # 在此调用 upsert_chunks 时传入 valid_to_seq > 0，
        # 检索时 list_visible 会排除 valid_to_seq > active_seq 的 chunk
        self.upsert_chunks(
            chunk_ids,
            scenario_id=scenario_id,
            source=source,
            kb_version=kb_version,
            valid_from_seq=valid_from_seq,
            valid_to_seq=valid_to_seq,
            file_path=file_path,
        )

    def list_visible(
        self,
        *,
        scenario_id: str,
        active_seq: int,
        source: str | None = None,
    ) -> list[ChunkVersionRecord]:
        """List chunks visible at a specific active version sequence.

        调用顺序：治理或版本管理入口 -> ChunkVersionIndex.list_visible()。
        """

        filters = [
            "scenario_id = :scenario_id",
            "valid_from_seq <= :active_seq",
            # valid_to_seq == 0 表示该 chunk 尚未被过期（仍在有效窗口内）；
            # valid_to_seq > active_seq 表示该 chunk 在此版本后仍有效
            "(valid_to_seq = 0 OR valid_to_seq > :active_seq)",
        ]
        params: dict[str, Any] = {"scenario_id": scenario_id, "active_seq": int(active_seq)}
        if source:
            filters.append("source = :source")
            params["source"] = source
        sql = f"""
        SELECT
            scenario_id, chunk_id, source, kb_version, file_path,
            valid_from_seq, valid_to_seq
        FROM {self.table_name}
        WHERE {' AND '.join(filters)}
        ORDER BY source ASC, chunk_id ASC
        """
        with self.engine.begin() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
        return [ChunkVersionRecord.from_row(row) for row in rows]
