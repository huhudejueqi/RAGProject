"""FAQ CSV 入库链路。FAQ 的 page_content 是标准问题，标准答案放在 metadata.answer。"""

from __future__ import annotations

import pandas as pd
from langchain_core.documents import Document

from qa_core.config.logging_config import get_logger
from qa_core.quality.faq import _resolve_csv_source
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import get_kb_version_store, version_metadata
from qa_core.indexing.source_normalization import normalize_faq_source
from qa_core.retrieval.factory import get_faq_store
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import stable_hash
logger = get_logger(__name__)

def faq_documents_from_csv(
    csv_path: str,
    kb_version: str | None = None,
    version_seq: int | None = None,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
) -> tuple[list[Document], list[str]]:
    """把 FAQ CSV 转换为可写入 Milvus 的问题文档。page_content=标准问题，metadata.answer=标准答案。

    调用顺序：入库脚本或索引服务 -> faq_documents_from_csv()。
    """
    scenario = resolve_scenario(scenario_id)
    data_scope = resolve_data_scope(tenant_id=tenant_id, dataset_id=dataset_id, visibility=visibility, user_roles=allowed_roles)
    version_meta = version_metadata(kb_version, scenario.scenario_id, version_seq=version_seq)
    # 原因： pandas 自动处理 BOM/编码推断/空值填充，而 csv.DictReader 只做逐行原始解析，遇到同名列合并或编码抖动需要额外编排才能达到同等健壮性
    data = pd.read_csv(csv_path, encoding="utf-8")
    docs: list[Document] = []
    ids: list[str] = []
    seen_ids: set[str] = set()
    for _, row in data.iterrows():
        # 原因： 中文客户 CSV 可能用中文列名（问题/答案）也可能用英文列名（question/answer），同时兼容两种 header 减少运维沟通成本
        question = str(row.get("问题") or row.get("question") or "").strip()
        answer = str(row.get("答案") or row.get("answer") or "").strip()
        subject = _resolve_csv_source(dict(row))
        # 跳过问题或答案缺失的行：FAQ 必须同时有标准问题和标准答案才有入库价值
        if not question or not answer:
            continue

        source = normalize_faq_source(subject, scenario=scenario, question=question)
        faq_id = stable_hash(scenario.scenario_id, kb_version or "", source, question)
        if faq_id in seen_ids:
            # 同一标准问题但答案不同，使用答案参与 hash，避免 id 冲突。
            faq_id = stable_hash(scenario.scenario_id, kb_version or "", source, question, answer)
        if faq_id in seen_ids:
            # 加入答案后仍然冲突（完全重复行），跳过重复记录
            continue
        seen_ids.add(faq_id)
        docs.append(
            Document(
                # page_content 仅存标准问题，答案放在 metadata 中，检索时用问题匹配
                # 召回后将答案作为上下文返回给用户
                page_content=question,
                metadata={
                    "faq_id": faq_id,
                    "scenario_id": scenario.scenario_id,
                    # FAQ 采用按版本快照重建模式；valid_from_seq/valid_to_seq 只是公共版本字段。
                    "source_type": "faq",
                    "record_type": "faq",
                    "versioning_mode": "snapshot",
                    "version_filter_mode": "kb_version_exact",
                    **data_scope.metadata(allowed_roles=allowed_roles),
                    "standard_question": question,
                    "answer": answer,
                    "source": source,
                    "subject_name": subject,
                    "status": "published",
                    **version_meta,
                },
            )
        )
        ids.append(faq_id)
    return docs, ids


def ingest_faq_csv(
    csv_path: str,
    *,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    kb_version: str | None = None,
    create_new_version: bool = False,
    description: str = "",
) -> int:
    """从 CSV 重新构建 FAQ 记录并写入 FAQ 混合集合。FAQ id 包含 kb_version，新版本不会覆盖旧版本。

    调用顺序：入库脚本或索引服务 -> ingest_faq_csv()。
    """
    scenario = resolve_scenario(scenario_id)
    version_store = get_kb_version_store(scenario.scenario_id)
    # 确保版本记录存在：如果 create_new=True 则自动生成新版本号；否则使用已有版本或 active 版本
    version = version_store.ensure_version(
        kb_version,
        create_new=create_new_version,
        description=description,
        created_by="ingest_faq_csv",
    )
    active_kb_version = version.kb_version
    docs, ids = faq_documents_from_csv(
        csv_path,
        active_kb_version,
        scenario_id=scenario.scenario_id,
        version_seq=version.version_seq,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        visibility=visibility,
        allowed_roles=allowed_roles,
    )
    store = get_faq_store(scenario.faq_collection)
    # 先删除再写入：FAQ 整体替换策略，确保旧版本 FAQ 不会与新版本 FAQ 残留混合
    store.delete_ids(ids)
    store.add_documents(docs, ids=ids)
    version_store.record_ingest_result(active_kb_version, content_type="faq", count=len(docs))
    logger.info("Ingested %s FAQ records from %s, kb_version: %s", len(docs), csv_path, active_kb_version)
    return len(docs)

