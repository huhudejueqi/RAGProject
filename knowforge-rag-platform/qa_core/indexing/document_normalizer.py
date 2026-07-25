"""入库文档元数据标准化。补充 source、scenario、kb_version、tenant、dataset 等元数据。"""

from __future__ import annotations
from pathlib import Path
from langchain_core.documents import Document
from qa_core.governance.data_scope import DataScope, resolve_data_scope
from qa_core.governance.kb_versions import version_metadata
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import file_fingerprint

def normalize_documents(
    documents: list[Document],
    file_path: Path,
    source: str,
    kb_version: str | None = None,
    scenario_id: str | None = None,
    version_seq: int | None = None,
    data_scope: DataScope | None = None,
    allowed_roles: list[str] | None = None,
) -> list[Document]:
    """为文档补充项目标准元数据（source/scenario_id/数据域/文件信息/doc_id/版本信息）。

    调用顺序：入库脚本或索引服务 -> normalize_documents()。
    """
    # 基于文件路径和内容生成稳定的 doc_id，同一文件多次入库 doc_id 不变，用于去重和增量检测
    doc_id = file_fingerprint(file_path)
    scenario = resolve_scenario(scenario_id)
    # 未显式传入 data_scope 时使用全局默认值（default tenant/dataset/public），
    # 确保所有文档至少有一个安全的数据域边界
    scope = data_scope or resolve_data_scope()
    version_meta = version_metadata(kb_version, scenario.scenario_id, version_seq=version_seq)
    normalized: list[Document] = []
    for index, doc in enumerate(documents):
        metadata = dict(doc.metadata)
        metadata.update(
            {
                "source": source,
                "scenario_id": scenario.scenario_id,
                # 文档采用引用式增量；在线检索按 version_seq 有效期窗口判断可见性。
                "source_type": "doc",
                "record_type": "doc_chunk",
                "versioning_mode": "reference_incremental",
                "version_filter_mode": "validity_window",
                **scope.metadata(allowed_roles=allowed_roles),
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_type": file_path.suffix.lower(),
                "doc_id": doc_id,
                # page_index 优先使用 loader 自带页码，fallback 到文档列表中的索引位置
                "page_index": metadata.get("page", index),
                # content_type 只覆盖空值：如果 loader 已标记 content_type（如 table_row），保持原值不变
                "content_type": metadata.get("content_type") or "text",
                **version_meta,
            }
        )
        normalized.append(Document(page_content=doc.page_content, metadata=metadata))
    return normalized
