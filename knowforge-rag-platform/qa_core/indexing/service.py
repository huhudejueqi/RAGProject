"""本地业务文档入库编排服务。

这个文件负责“离线入库链路”，不参与在线问答时的实时生成。它把一个业务场景目录下的
本地资料按固定流程写入 Milvus：解析场景配置 → 确认数据隔离范围 → 确认知识库版本 →
加载文件 → 标准化元数据 → 切分 chunk → 写入向量库 → 更新本地索引清单。

这里单独成一个 service 文件，是为了把入库流程和在线 QAService 分开：
- 在线问答只负责检索、重排、Prompt 和流式返回；
- 离线入库只负责资料治理、增量构建、版本记录和 Milvus 写入。
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.governance.chunk_versions import ChunkVersionIndex
from qa_core.governance.data_scope import resolve_data_scope
from qa_core.governance.kb_versions import get_kb_version_store
from qa_core.indexing.chunking import split_documents
from qa_core.indexing.document_loaders import get_document_loader_spec, load_file
from qa_core.indexing.document_normalizer import normalize_documents
from qa_core.indexing.manifest import IndexManifest
from qa_core.retrieval.factory import get_doc_store
from qa_core.scenarios.registry import resolve_scenario
from qa_core.utils import file_fingerprint, normalize_source_from_path
from qa_core.knowledge_graph.pipeline import run_knowledge_graph_pipeline


logger = get_logger(__name__)


@dataclass(frozen=True)
class DocumentIngestContext:
    """单次目录入库时所有文件共享的上下文。

    调用顺序：入库脚本或索引服务 -> DocumentIngestContext。
    """

    source: str
    kb_version: str
    kb_version_seq: int
    scenario: Any
    data_scope: Any
    allowed_roles: list[str] | None
    doc_store: Any
    manifest: IndexManifest
    chunk_index: ChunkVersionIndex
    force: bool
    incremental_base_kb_version: str | None = None
    incremental_base_version_seq: int = 0

    @property
    def scenario_id(self) -> str:
        """返回当前上下文的场景标识。

        调用顺序：入库脚本或索引服务 -> DocumentIngestContext.scenario_id()。
        """
        return self.scenario.scenario_id


@dataclass(frozen=True)
class FileIngestResult:
    """单个文件入库后的统计结果。

    调用顺序：入库脚本或索引服务 -> FileIngestResult。
    """

    reembedded_chunks: int = 0
    skipped: bool = False
    reused_chunks: int = 0
    expired_chunks: int = 0

    @property
    def total_chunks(self) -> int:
        """返回该文件入库的 chunk 总数（重新写入 + 引用复用）。

        调用顺序：入库脚本或索引服务 -> FileIngestResult.total_chunks()。
        """
        return self.reembedded_chunks + self.reused_chunks


@dataclass
class DirectoryIngestStats:
    """目录入库统计，用于写入知识库版本 stats。

    调用顺序：入库脚本或索引服务 -> DirectoryIngestStats。
    """

    reembedded_chunks: int = 0
    reused_chunks: int = 0
    expired_chunks: int = 0
    skipped_files: int = 0

    @property
    def total_chunks(self) -> int:
        """返回该目录入库的 chunk 总数（重新写入 + 引用复用）。

        调用顺序：入库脚本或索引服务 -> DirectoryIngestStats.total_chunks()。
        """
        return self.reembedded_chunks + self.reused_chunks

    def add(self, result: FileIngestResult) -> None:
        """累加一个文件入库结果到目录统计中。

        参数：
            result: 单个文件的入库结果，含各类型 chunk 计数和跳过标记。

        调用顺序：入库脚本或索引服务 -> DirectoryIngestStats.add()。
        """
        if result.skipped:
            self.skipped_files += 1
        self.reembedded_chunks += result.reembedded_chunks
        self.reused_chunks += result.reused_chunks
        self.expired_chunks += result.expired_chunks

    def as_version_stats(self, incremental_base_kb_version: str | None) -> dict[str, int | str]:
        """将入库统计转换为版本记录用的 stats 字典。

        参数：
            incremental_base_kb_version: 增量基准版本号，为空时对应字段为空字符串。

        返回：
            含重新写入数、复用数、失效数、跳过文件数和基准版本的字典。

        调用顺序：入库脚本或索引服务 -> DirectoryIngestStats.as_version_stats()。
        """
        return {
            "last_doc_reembedded_count": self.reembedded_chunks,
            "last_doc_reused_count": self.reused_chunks,
            "last_doc_expired_count": self.expired_chunks,
            "last_doc_skipped_file_count": self.skipped_files,
            "last_doc_incremental_base_kb_version": incremental_base_kb_version or "",
        }


def _walk_files(root: Path):
    """递归产出目录中的文件路径。

    调用顺序：入库脚本或索引服务 -> _walk_files()。
    """
    for current_root, _, files in os.walk(root):
        for file_name in files:
            yield Path(current_root) / file_name


def _manifest_matches_current_settings(record, fingerprint: str, settings) -> bool:
    """判断 manifest 记录是否仍可用于当前文件、embedding 模型和 chunk schema。

    这里是严格等值匹配，不做“字段种类没少即可复用”的宽松兼容：
      1. fingerprint 一致，表示本地文件未变化。
      2. embedding_model_version 一致，表示旧向量仍处于同一向量空间。
      3. chunk_schema_version 一致，表示切分规则和 chunk metadata 契约未变化。

    `valid_from_seq / valid_to_seq` 属于 chunk metadata 契约。引入或调整这类会影响
    检索过滤语义的字段时，应提升 CHUNK_SCHEMA_VERSION，让旧 manifest 自动失配并重建。

    调用顺序：入库脚本或索引服务 -> _manifest_matches_current_settings()。
    """
    return bool(
        record
        and record.fingerprint == fingerprint
        and record.embedding_model_version == settings.embedding_model_version
        and record.chunk_schema_version == settings.chunk_schema_version
    )


def _record_manifest(
    context: DocumentIngestContext,
    path: Path,
    fingerprint: str,
    chunk_ids: list[str],
    settings,
) -> None:
    """把一次成功写入或跨版本复用结果写回 MySQL manifest。

    调用顺序：入库脚本或索引服务 -> _record_manifest()。
    """
    context.manifest.update(
        context.source,
        path,
        fingerprint,
        chunk_ids,
        scenario_id=context.scenario_id,
        kb_version=context.kb_version,
        embedding_model_version=settings.embedding_model_version,
        chunk_schema_version=settings.chunk_schema_version,
    )


def _find_reusable_base_record(
    context: DocumentIngestContext,
    path: Path,
    fingerprint: str,
    settings,
):
    """在基准版本中寻找可复用的未变化文件记录。

    调用顺序：入库脚本或索引服务 -> _find_reusable_base_record()。
    """
    if not context.incremental_base_kb_version or context.force:
        return None
    base_existing = context.manifest.get(
        context.source,
        path,
        context.incremental_base_kb_version,
        context.scenario_id,
    )
    return base_existing if _manifest_matches_current_settings(base_existing, fingerprint, settings) else None


def _find_base_record(context: DocumentIngestContext, path: Path):
    """读取基线版本里同一路径的 manifest 记录，用于引用复用或失效旧 chunk。

    调用顺序：入库脚本或索引服务 -> _find_base_record()。
    """
    if not context.incremental_base_kb_version or context.force:
        return None
    return context.manifest.get(
        context.source,
        path,
        context.incremental_base_kb_version,
        context.scenario_id,
    )


def _reference_base_version(
    context: DocumentIngestContext,
    path: Path,
    fingerprint: str,
    base_record,
    settings,
) -> FileIngestResult:
    """引用基准版本中未变化文件的 chunk，不再复制 Milvus 行。

    调用顺序：入库脚本或索引服务 -> _reference_base_version()。
    """
    if context.incremental_base_version_seq:
        context.doc_store.ensure_documents_validity(
            base_record.chunk_ids,
            valid_from_seq=context.incremental_base_version_seq,
        )
        context.chunk_index.ensure_chunks_validity(
            base_record.chunk_ids,
            scenario_id=context.scenario_id,
            source=context.source,
            kb_version=base_record.kb_version,
            valid_from_seq=context.incremental_base_version_seq,
            file_path=base_record.path,
        )
    _record_manifest(context, path, fingerprint, base_record.chunk_ids, settings)
    return FileIngestResult(reused_chunks=len(base_record.chunk_ids))


def _expire_base_record_for_target(context: DocumentIngestContext, base_record) -> int:
    """让基线 chunk 从目标版本开始失效。

    调用顺序：入库脚本或索引服务 -> _expire_base_record_for_target()。
    """
    if not base_record or not base_record.chunk_ids:
        return 0
    expired = context.doc_store.expire_documents_for_version(
        base_record.chunk_ids,
        valid_to_seq=context.kb_version_seq,
        valid_from_seq=context.incremental_base_version_seq or None,
    )
    context.chunk_index.expire_chunks(
        base_record.chunk_ids,
        scenario_id=context.scenario_id,
        source=context.source,
        kb_version=base_record.kb_version,
        valid_from_seq=context.incremental_base_version_seq,
        valid_to_seq=context.kb_version_seq,
        file_path=base_record.path,
    )
    return expired


def _delete_existing_target_chunks(context: DocumentIngestContext, existing) -> None:
    """删除目标版本自己写入的旧 chunk，保留被目标 manifest 引用的基线 chunk。

    调用顺序：入库脚本或索引服务 -> _delete_existing_target_chunks()。
    """
    if not existing or not existing.chunk_ids:
        return
    if hasattr(context.doc_store, "delete_ids_for_kb_version"):
        context.doc_store.delete_ids_for_kb_version(existing.chunk_ids, context.kb_version)
        return
    context.doc_store.delete_ids(existing.chunk_ids)


def _rebuild_file_chunks(
    context: DocumentIngestContext,
    path: Path,
    fingerprint: str,
    existing,
    settings,
    *,
    _chunk_collector: list | None = None,
    expired_chunks: int = 0,
) -> FileIngestResult:
    """重新加载、标准化、切分并写入一个已变化或新增的文件。

    调用顺序：入库脚本或索引服务 -> _rebuild_file_chunks()。
    """
    _delete_existing_target_chunks(context, existing)
    docs = normalize_documents(
        load_file(path),
        path,
        context.source,
        context.kb_version,
        context.scenario_id,
        context.kb_version_seq,
        context.data_scope,
        context.allowed_roles,
    )
    chunks, ids = split_documents(docs)
    if not chunks:
        return FileIngestResult()
    context.doc_store.add_documents(chunks, ids=ids)
    context.chunk_index.upsert_chunks(
        ids,
        scenario_id=context.scenario_id,
        source=context.source,
        kb_version=context.kb_version,
        valid_from_seq=context.kb_version_seq,
        file_path=str(path.resolve()),
    )
    if _chunk_collector is not None:
        _chunk_collector.extend(chunks)
    _record_manifest(context, path, fingerprint, ids, settings)
    return FileIngestResult(reembedded_chunks=len(chunks), expired_chunks=expired_chunks)


def _ingest_single_file(path: Path, context: DocumentIngestContext, *, _chunk_collector: list | None = None) -> FileIngestResult:
    """处理单个文件的增量入库。

    这个函数是 `ingest_directory()` 的最小执行单元。之所以拆出来，是因为“目录遍历”
    和“单文件是否需要重建”是两层不同逻辑：目录层只负责找文件，文件层负责判断是否跳过、
    是否删除旧 chunk、是否重新加载和写入。

    执行流程：
      1. 校验文件类型是否有 LangChain Loader 支持，不支持就直接报错。
      2. 根据文件路径、修改时间、大小生成 fingerprint，用于判断文件是否变化。
      3. 从 manifest 中查找同一场景、同一版本、同一 source、同一路径的旧入库记录。
      4. 如果文件指纹、embedding 版本、chunk schema 都没变，并且没有强制重建，则跳过。
      5. 如果文件已变化或 schema 升级，先删除旧 chunk，避免 Milvus 中同一文件新旧内容并存。
      6. 通过 LangChain Loader 读取文件，再补齐场景、版本、数据域等 metadata。
      7. 调用统一切分器生成 chunk 和稳定 chunk_id。
      8. 将 chunk 写入 Milvus，并把新 chunk_id 记录回 manifest。

    返回值：
      - reembedded_chunks：本次重新 embedding 并写入的 chunk 数。
      - skipped：是否因为目标版本内文件未变化而跳过。
      - reused_chunks：从基准版本复制复用的 chunk 数。
    """
    if get_document_loader_spec(path) is None:
        raise ValueError(f"不支持的文档类型：{path}")
    fingerprint = file_fingerprint(path)
    settings = get_settings()
    existing = context.manifest.get(context.source, path, context.kb_version, context.scenario_id)
    # 文件指纹 + embedding/chunk schema 均未变 → 跳过入库，实现同版本增量式索引更新
    if not context.force and _manifest_matches_current_settings(existing, fingerprint, settings):
        return FileIngestResult(skipped=True)

    base_record = _find_base_record(context, path)
    if not existing and base_record and _manifest_matches_current_settings(base_record, fingerprint, settings):
        return _reference_base_version(context, path, fingerprint, base_record, settings)

    expired_chunks = 0
    if base_record:
        expired_chunks = _expire_base_record_for_target(context, base_record)

    # 文件已变更或 schema 升级，清理旧 chunk 后重新入库，防止向量数据版本混乱
    return _rebuild_file_chunks(context, path, fingerprint, existing, settings, expired_chunks=expired_chunks, _chunk_collector=_chunk_collector)


def _expire_missing_base_records(context: DocumentIngestContext, seen_paths: set[str]) -> int:
    """目标版本中已删除的文件不再复制，只让基线 chunk 从目标版本开始不可见。

    调用顺序：入库脚本或索引服务 -> _expire_missing_base_records()。
    """
    if not context.incremental_base_kb_version:
        return 0
    expired_chunks = 0
    base_records = context.manifest.iter_records(
        scenario_id=context.scenario_id,
        source=context.source,
        kb_version=context.incremental_base_kb_version,
    )
    for record in base_records:
        if str(Path(record.path).resolve()) in seen_paths:
            continue
        expired_chunks += _expire_base_record_for_target(context, record)
    return expired_chunks


def ingest_directory(
    directory_path: str,
    source: str | None = None,
    *,
    scenario_id: str | None = None,
    tenant_id: str | None = None,
    dataset_id: str | None = None,
    visibility: str | None = None,
    allowed_roles: list[str] | None = None,
    force: bool = False,
    kb_version: str | None = None,
    create_new_version: bool = False,
    description: str = "",
    incremental_base_kb_version: str | None = None,
) -> int:
    """把某个目录下的业务文档增量写入 Milvus。

    这是文档入库的主入口，通常由脚本调用，例如重建某个场景的知识库版本时会走这里。
    它不负责 FAQ CSV，FAQ 有单独的 `faq_ingestion.py`；这里专注处理普通业务文档、
    表格行、OCR 后的文本等“文档型资料”。

    主要职责：
      1. 解析当前业务场景，拿到 doc_collection、valid_sources、版本清单路径等配置。
      2. 构建 DataScope，把 tenant/dataset/visibility/roles 写入 metadata，支持隔离检索。
      3. 校验 source 必须属于当前场景的 valid_sources，防止跨场景数据写错集合。
      4. 确认或创建知识库版本，新旧版本可以并存，线上只检索 active 版本。
      5. 递归遍历目录，对每个文件调用 `_ingest_single_file()` 做增量判断和写入。
      6. 保存 manifest，让下次入库可以跳过未变化文件，并能删除旧 chunk。
      7. 记录本次入库统计；版本发布由 rebuild_kb_version.py 的质量门禁收口。

    参数说明：
      - directory_path：要入库的目录。
      - source：业务分类；不传时从目录名推断，例如 `finance_data` 推断为 `finance`。
      - scenario_id：目标业务场景，例如 `enterprise_knowledge`。
      - tenant_id/dataset_id/visibility/allowed_roles：数据隔离字段，会进入 Milvus metadata。
      - force：是否忽略 fingerprint，强制重建所有文件。
      - incremental_base_kb_version：跨版本增量构建的基准版本；未变化文件会引用旧 chunk，不复制 Milvus 行。

    返回：
      实际写入 Milvus 的 chunk 总数，不包含被增量跳过的文件。

    调用顺序：入库脚本或索引服务 -> ingest_directory()。
    """
    scenario = resolve_scenario(scenario_id)
    data_scope = resolve_data_scope(tenant_id=tenant_id, dataset_id=dataset_id, visibility=visibility, user_roles=allowed_roles)
    root = Path(directory_path)
    # 未显式传 source 时，从当前目录名推断业务分类，例如 finance_data → finance。
    # 推断结果仍要经过场景 valid_sources 校验，防止目录名写错后把数据写进错误分类。
    resolved_source = source or normalize_source_from_path(root)
    if resolved_source not in scenario.valid_sources:
        raise ValueError(f"无效的业务分类：{resolved_source}，当前场景支持：{scenario.valid_sources}")
    version_store = get_kb_version_store(scenario.scenario_id)
    version = version_store.ensure_version(
        kb_version,
        create_new=create_new_version,
        description=description,
        created_by="ingest_directory",
    )
    active_kb_version = version.kb_version
    incremental_base_version_seq = 0
    if incremental_base_kb_version:
        base_version = version_store.get(incremental_base_kb_version)
        if base_version is None:
            raise ValueError(f"增量基准版本不存在：{incremental_base_kb_version}")
        incremental_base_version_seq = base_version.version_seq
    manifest = IndexManifest()
    chunk_index = ChunkVersionIndex()
    doc_store = get_doc_store(scenario.doc_collection)
    context = DocumentIngestContext(
        source=resolved_source,
        kb_version=active_kb_version,
        kb_version_seq=version.version_seq,
        scenario=scenario,
        data_scope=data_scope,
        allowed_roles=allowed_roles,
        doc_store=doc_store,
        manifest=manifest,
        chunk_index=chunk_index,
        force=force,
        incremental_base_kb_version=incremental_base_kb_version,
        incremental_base_version_seq=incremental_base_version_seq,
    )
    stats = DirectoryIngestStats()
    seen_paths: set[str] = set()
    _all_chunks: list = []  # 收集新建 chunk 用于知识图谱构建
    for path in _walk_files(root):
        seen_paths.add(str(path.resolve()))
        stats.add(_ingest_single_file(path, context, _chunk_collector=_all_chunks))
    stats.expired_chunks += _expire_missing_base_records(context, seen_paths)
    # 在版本清单中记录本次入库统计
    version_store.record_ingest_result(
        active_kb_version,
        content_type="doc",
        count=stats.total_chunks,
        source=resolved_source,
        extra_stats=stats.as_version_stats(incremental_base_kb_version),
    )
    logger.info(
        "文档入库完成：目标版本 chunk=%s，重新写入=%s，引用复用=%s，失效旧chunk=%s，目录=%s，跳过未变化文件=%s，kb_version=%s",
        stats.total_chunks,
        stats.reembedded_chunks,
        stats.reused_chunks,
        stats.expired_chunks,
        directory_path,
        stats.skipped_files,
        active_kb_version,
    )
    # ── 知识图谱构建（异步触发） ──
    if _all_chunks:
        try:
            import asyncio
            kg_result = asyncio.run(run_knowledge_graph_pipeline(
                _all_chunks,
                kb_version=active_kb_version,
            ))
            logger.info(
                "知识图谱构建完成: %d 实体, %d 关系, %d 社群",
                kg_result.entities_extracted,
                kg_result.relationships_extracted,
                kg_result.communities_detected,
            )
            if kg_result.errors:
                logger.warning("知识图谱构建部分失败: %s", kg_result.errors)
        except Exception as e:
            logger.warning("知识图谱构建异常: %s", e)
    return stats.total_chunks



