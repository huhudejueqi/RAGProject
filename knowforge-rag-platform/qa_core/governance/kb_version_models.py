"""知识库版本的数据结构和字段序列化。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

from qa_core.common import utc_now

KB_VERSION_STATUS_STAGED = "STAGED"
KB_VERSION_STATUS_ACTIVE = "ACTIVE"
KB_VERSION_STATUS_ARCHIVED = "ARCHIVED"
KB_VERSION_SELECT_COLUMNS = (
    "scenario_id, kb_version, version_seq, status, description, created_at, "
    "activated_at, archived_at, doc_collection, faq_collection, "
    "embedding_model_version, reranker_model_version, chunk_schema_version, "
    "created_by, sources_json, stats_json"
)


def json_dumps(value: Any) -> str:
    """按项目统一方式序列化 JSON 字段。

    调用顺序：治理或版本管理入口 -> json_dumps()。
    """
    return json.dumps(value or {}, ensure_ascii=False, sort_keys=True)


def json_loads_dict(value: Any) -> dict[str, Any]:
    """把数据库 JSON/TEXT 字段恢复为 dict，坏数据按空字典处理。

    调用顺序：治理或版本管理入口 -> json_loads_dict()。
    """
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def json_loads_list(value: Any) -> list[str]:
    """把数据库 JSON/TEXT 字段恢复为字符串列表。

    调用顺序：治理或版本管理入口 -> json_loads_list()。
    """
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        payload = json.loads(str(value))
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [str(item) for item in payload]


@dataclass
class KnowledgeBaseVersion:
    """可检索知识库版本的元数据。

    调用顺序：治理或版本管理入口 -> KnowledgeBaseVersion。
    """

    kb_version: str
    scenario_id: str = ""
    version_seq: int = 0
    status: str = KB_VERSION_STATUS_STAGED
    description: str = ""
    created_at: str = field(default_factory=utc_now)
    activated_at: str | None = None
    archived_at: str | None = None
    doc_collection: str = ""
    faq_collection: str = ""
    embedding_model_version: str = ""
    reranker_model_version: str = ""
    chunk_schema_version: str = ""
    created_by: str = "local"
    sources: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "KnowledgeBaseVersion":
        """从 dict 恢复版本对象，老记录缺失的字段自动使用默认值。

        调用顺序：治理或版本管理入口 -> KnowledgeBaseVersion.from_dict()。
        """
        fields = cls.__dataclass_fields__
        data = {name: payload.get(name) for name in fields if name in payload}
        version = cls(**data)
        if version.sources is None:
            version.sources = []
        if version.stats is None:
            version.stats = {}
        version.version_seq = int(version.version_seq or 0)
        return version

    @classmethod
    def from_row(cls, row: Any) -> "KnowledgeBaseVersion":
        """从 SQLAlchemy RowMapping 恢复版本对象。

        调用顺序：治理或版本管理入口 -> KnowledgeBaseVersion.from_row()。
        """
        payload = dict(row)
        payload["sources"] = json_loads_list(payload.get("sources_json"))
        payload["stats"] = json_loads_dict(payload.get("stats_json"))
        payload.pop("sources_json", None)
        payload.pop("stats_json", None)
        return cls.from_dict(payload)

    def as_dict(self) -> dict[str, Any]:
        """返回可 JSON 序列化的版本信息。

        调用顺序：治理或版本管理入口 -> KnowledgeBaseVersion.as_dict()。
        """
        return asdict(self)
