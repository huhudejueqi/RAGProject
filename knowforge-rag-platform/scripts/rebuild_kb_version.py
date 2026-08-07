# -*- coding: utf-8 -*-
"""构建单个业务场景的完整知识库版本。

业务目标：
    把 FAQ 和文档写入同一个 kb_version，并在质量门禁通过后按需激活。

通用发布流程：
    1. 解析参数并校验冲突选项
    2. 解析业务场景，必要时重建 Milvus collection
    3. 创建或复用目标 kb_version
    4. 可选解析跨版本增量基准
    5. FAQ 入库
    6. 文档入库
    7. 生成质量报告；如果要激活版本，则必须执行质量门禁
    8. 按需激活版本并输出结果

增量入库流程：
    1. 选择业务场景，例如 enterprise_knowledge。
    2. 创建新的目标 kb_version，新版本先保持 STAGED，不影响线上查询。
    3. 用 --incremental-from active 或显式旧 kb_version 确定增量基准版本。
    4. 在新版本 stats 中记录 incremental_base_kb_version，方便追溯。
    5. FAQ 仍然按新版本重建；FAQ 数量小且高置信直出口径要求更高，不做跨版本引用。
    6. 文档逐文件计算 fingerprint，并同时检查 embedding_model_version 和 chunk_schema_version。
    7. 文件未变化时，从基准版本 manifest 找到旧 chunk_ids，目标版本 manifest 直接引用这些
       chunk，不复制 Milvus 行、不重新 embedding。
    8. 文件新增、内容变化、模型变化或切分策略变化时，重新执行 load_file()、
       normalize_documents()、split_documents()、add_documents()，再更新 manifest。
    9. 文件删除时给旧 chunk 写 valid_to_seq，让它从目标版本开始不可见。
    10. 生成目标版本的质量报告，检查文件解析、FAQ、chunk 和 FAQ/正文冲突。
    11. 执行质量门禁；失败则不激活，旧 active 继续服务。
    12. 门禁通过且传入 --activate 时切换 MySQL active 指针，文档检索按 active version_seq
        解释有效期视图：valid_from_seq <= active_seq 且未失效。

关键边界：
    增量入库不是线上查询时拼接多个 kb_version，而是用 version_seq 解释同一个 collection
    中 chunk 的有效期视图。未变化 chunk 不复制，变化或删除通过 valid_to_seq 收口。
"""

from __future__ import annotations

# ── 标准库 ──
# argparse: 命令行参数解析
import argparse
# sys: 系统功能（sys.path 修改 + sys.exit 退出码）
import sys
# pathlib.Path: 文件路径操作
from pathlib import Path
# typing.Any: 任意类型
from typing import Any

# ── 路径设置 ──
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── 导入 Milvus SDK ──
# MilvusClient: PyMilvus 客户端（用于 Collection 的 drop 操作）
from pymilvus import MilvusClient

# ── 导入核心模块 ──
# get_kb_version_store: 获取知识库版本 Store（MySQL 中的版本控制面）
from qa_core.governance.kb_versions import get_kb_version_store
# ingest_faq_csv: FAQ CSV 入库（写入 FAQ collection）
from qa_core.indexing.faq_ingestion import ingest_faq_csv
# ingest_directory: 目录批量入库（逐文件 load → normalize → split → add）
from qa_core.indexing.service import ingest_directory
# build_ingestion_quality_report: 生成入库质量报告
# save_ingestion_quality_report: 保存入库质量报告到文件
from qa_core.quality.ingestion import build_ingestion_quality_report, save_ingestion_quality_report
# ensure_milvus_database: 确保 Milvus database 存在
# langchain_connection_args: 生成 langchain-milvus 兼容的连接参数
from qa_core.retrieval.milvus_compat import ensure_milvus_database, langchain_connection_args
# resolve_scenario: 解析业务场景配置
from qa_core.scenarios.registry import resolve_scenario
from qa_core.storage.bootstrap import bootstrap_mysql_schema
# IngestionQualityThresholds: 入库质量门禁阈值
# evaluate_report_against_gate: 用阈值判断入库质量报告是否通过
from scripts.check_ingestion_quality_gate import IngestionQualityThresholds, evaluate_report_against_gate


def build_parser() -> argparse.ArgumentParser:
    """构造命令行参数。

    调用顺序：命令行入口 -> build_parser()。
    """
    parser = argparse.ArgumentParser(description="Rebuild one scenario into a complete knowledge base version.")
    # 文档数据根目录：覆盖场景配置里的 data_root，用于批量加载业务文档。
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Root data directory. Defaults to the selected scenario data_root.",
    )
    # FAQ CSV 路径：覆盖场景配置里的 faq_csv_path。
    parser.add_argument(
        "--faq-csv",
        default=None,
        help="FAQ CSV path. Defaults to the selected scenario faq_csv_path.",
    )
    # 场景 ID：确定要构建哪个业务知识库，默认取 ACTIVE_SCENARIO_ID。
    parser.add_argument("--scenario", default=None, help="Business scenario id. Defaults to ACTIVE_SCENARIO_ID.")
    # 目标版本号：指定已有 kb_version；不新建时复用该版本。
    parser.add_argument("--kb-version", default=None, help="Explicit knowledge base version id.")
    # 新建版本：创建 STAGED 新版本，不覆盖已有版本。
    parser.add_argument("--new-version", action="store_true", help="Create a new staged version before ingest.")
    # 强制重建：忽略文件指纹，重新加载、切分和 embedding。
    parser.add_argument("--force", action="store_true", help="Rebuild files even when fingerprint is unchanged.")
    # 增量基准版本：未变化文档复用基准版本 chunk，变化/删除文档按有效期失效。
    parser.add_argument(
        "--incremental-from",
        default=None,
        help=(
            "Cross-version incremental document build base. Use 'active' to copy unchanged chunks "
            "from the current active version, or pass an explicit kb_version. FAQ is still rebuilt."
        ),
    )
    # 跳过 FAQ 入库：只处理文档，适合单独补文档。
    parser.add_argument("--skip-faq", action="store_true", help="Skip FAQ ingest.")
    # 跳过文档入库：只处理 FAQ，适合单独补 FAQ。
    parser.add_argument("--skip-docs", action="store_true", help="Skip document ingest.")
    # 重置 collection：drop 当前场景 FAQ/Doc Milvus collection，schema 变更时使用。
    parser.add_argument(
        "--reset-collections",
        action="store_true",
        help=(
            "Drop the selected scenario FAQ and document Milvus collections before ingest. "
            "Use this when schema changed, especially when migrating to BM25 BuiltInFunction hybrid search."
        ),
    )
    # 跳过质量报告：仅允许 STAGED 构建，不能用于 --activate。
    parser.add_argument("--skip-quality-report", action="store_true", help="Skip ingestion quality report generation. Only allowed for staged builds.")
    # 质量门禁：严格检查入库质量，超过阈值则失败；激活版本时自动开启。
    parser.add_argument("--quality-gate", action="store_true", help="Run strict ingestion quality gate. Activation always enables it.")
    # 激活版本：质量门禁通过后切换 active 指针，线上检索立即使用新版本。
    parser.add_argument("--activate", action="store_true", help="Activate this version after successful ingest.")
    # 版本描述：写入 kb_versions.description，用于发布审计。
    parser.add_argument("--description", default="", help="Human readable version description.")
    # 租户 ID：写入 Milvus metadata，用于租户级数据隔离。
    parser.add_argument("--tenant-id", default=None, help="Tenant/org id written into metadata. Defaults to default.")
    # 数据集 ID：写入 Milvus metadata，用于数据集级数据隔离。
    parser.add_argument("--dataset-id", default=None, help="Dataset id written into metadata. Defaults to default.")
    # 可见级别：public/internal/private，控制谁能检索到这批数据。
    parser.add_argument("--visibility", default=None, help="Visibility written into metadata: public/internal/private.")
    # 允许角色：可重复传入，写入 metadata 做角色级数据隔离。
    parser.add_argument("--allowed-role", action="append", default=None, help="Role allowed to retrieve this data. Can repeat.")

    # 最大解析失败文件数：超过 0 个时质量门禁失败。
    parser.add_argument("--max-failed-files", type=int, default=0, help="Quality gate threshold.")
    # 最大不支持文件数：文档类型没有对应 loader 时计入。
    parser.add_argument("--max-unsupported-files", type=int, default=0, help="Quality gate threshold.")
    # 最大空文件数：文件加载后没有正文时计入。
    parser.add_argument("--max-empty-files", type=int, default=0, help="Quality gate threshold.")
    # 最大低质量文档问题数：切分/元数据质量不足时计入。
    parser.add_argument("--max-low-quality-issues", type=int, default=0, help="Quality gate threshold.")
    # 最大重复 chunk 数：同一内容被重复写入文档集合时计入。
    parser.add_argument("--max-duplicate-chunks", type=int, default=0, help="Quality gate threshold.")
    # 最大空 FAQ 问题数：CSV 中问题缺失时计入。
    parser.add_argument("--max-empty-faq-questions", type=int, default=0, help="Quality gate threshold.")
    # 最大空 FAQ 答案数：CSV 中答案缺失时计入。
    parser.add_argument("--max-empty-faq-answers", type=int, default=0, help="Quality gate threshold.")
    # 最大重复 FAQ 问题数：同一标准问题重复出现时计入。
    parser.add_argument("--max-duplicate-faq-questions", type=int, default=0, help="Quality gate threshold.")
    # 最大非法 FAQ source 数：FAQ 业务分类不在场景白名单时计入。
    parser.add_argument("--max-invalid-faq-sources", type=int, default=0, help="Quality gate threshold.")
    # 最大 FAQ 与文档冲突数：FAQ 标准口径和文档内容冲突时计入。
    parser.add_argument("--max-faq-document-conflicts", type=int, default=0, help="Quality gate threshold.")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """校验会破坏业务语义的参数组合。

    调用顺序：命令行入口 -> validate_args()。
    """
    if args.activate:
        args.quality_gate = True
    if args.activate and args.skip_quality_report:
        parser.error("--activate requires quality report and quality gate; remove --skip-quality-report.")
    if args.quality_gate and args.skip_quality_report:
        parser.error("--quality-gate requires quality report generation; remove --skip-quality-report.")
    if args.incremental_from and args.reset_collections:
        parser.error("--incremental-from cannot be used with --reset-collections because old vectors must be copied.")
    if args.incremental_from and args.force:
        parser.error("--incremental-from cannot be used with --force; force means rebuild all documents.")
    if args.incremental_from and not (args.new_version or args.kb_version):
        parser.error("--incremental-from requires --new-version or an explicit --kb-version target.")


def reset_collections_if_requested(args: argparse.Namespace, scenario: Any) -> None:
    """按需删除当前场景的 FAQ/Doc collection。

    调用顺序：命令行入口 -> reset_collections_if_requested()。
    """
    if not args.reset_collections:
        return

    ensure_milvus_database()
    client = MilvusClient(**langchain_connection_args())
    for collection_name in sorted({scenario.faq_collection, scenario.doc_collection}):
        if client.has_collection(collection_name):
            client.drop_collection(collection_name)
            print(f"Dropped Milvus collection for schema reset: {collection_name}")


def ensure_target_version(args: argparse.Namespace, scenario: Any):
    """创建或复用本次入库的目标版本。

    调用顺序：命令行入口 -> ensure_target_version()。
    """
    version_store = get_kb_version_store(scenario.scenario_id)
    create_new = args.new_version or (not args.kb_version and not bool(version_store.active_version_candidate()))
    version = version_store.ensure_version(
        args.kb_version,
        create_new=create_new,
        description=args.description,
        created_by="rebuild_kb_version",
    )
    return version_store, version


def resolve_incremental_base(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    version_store: Any,
    target_kb_version: str,
) -> str | None:
    """解析跨版本增量构建的基准版本。

    调用顺序：命令行入口 -> resolve_incremental_base()。
    """
    if not args.incremental_from:
        return None

    requested_base = args.incremental_from.strip()
    if requested_base.lower() == "active":
        base_kb_version = version_store.resolve_active_version()
    else:
        base_kb_version = version_store.resolve_active_version(requested_base)

    if base_kb_version == target_kb_version:
        parser.error("--incremental-from must point to a different base version than the target kb_version.")

    version_store.record_incremental_base(target_kb_version, base_kb_version)
    return base_kb_version


def ingest_faq(args: argparse.Namespace, scenario: Any, kb_version: str) -> int:
    """把 FAQ CSV 写入目标版本。

    调用顺序：命令行入口 -> ingest_faq()。
    """
    if args.skip_faq:
        return 0

    return ingest_faq_csv(
        args.faq_csv or scenario.faq_csv_path,
        scenario_id=scenario.scenario_id,
        tenant_id=args.tenant_id,
        dataset_id=args.dataset_id,
        visibility=args.visibility,
        allowed_roles=args.allowed_role,
        kb_version=kb_version,
        create_new_version=False,
        description=args.description,
    )


def ingest_documents(
    args: argparse.Namespace,
    scenario: Any,
    kb_version: str,
    incremental_base_kb_version: str | None,
) -> int:
    """把当前场景全部 source 目录写入目标版本。

    调用顺序：命令行入口 -> ingest_documents()。
    """
    if args.skip_docs:
        return 0

    total_chunks = 0
    root = Path(args.data_dir or scenario.data_root)
    for source in scenario.valid_sources:
        source_dir = root / f"{source}_data"
        if not source_dir.exists():
            continue
        total_chunks += ingest_directory(
            str(source_dir),
            source=source,
            scenario_id=scenario.scenario_id,
            tenant_id=args.tenant_id,
            dataset_id=args.dataset_id,
            visibility=args.visibility,
            allowed_roles=args.allowed_role,
            force=args.force,
            kb_version=kb_version,
            create_new_version=False,
            description=args.description,
            incremental_base_kb_version=incremental_base_kb_version,
        )
    return total_chunks


def quality_thresholds_from_args(args: argparse.Namespace) -> IngestionQualityThresholds:
    """从命令行参数构造入库质量门禁阈值。

    调用顺序：命令行入口 -> quality_thresholds_from_args()。
    """
    return IngestionQualityThresholds(
        max_failed_files=args.max_failed_files,
        max_unsupported_files=args.max_unsupported_files,
        max_empty_files=args.max_empty_files,
        max_low_quality_issues=args.max_low_quality_issues,
        max_duplicate_chunks=args.max_duplicate_chunks,
        max_empty_faq_questions=args.max_empty_faq_questions,
        max_empty_faq_answers=args.max_empty_faq_answers,
        max_duplicate_faq_questions=args.max_duplicate_faq_questions,
        max_invalid_faq_sources=args.max_invalid_faq_sources,
        max_faq_document_conflicts=args.max_faq_document_conflicts,
    )


def build_quality_report(
    args: argparse.Namespace,
    scenario: Any,
    kb_version: str,
    *,
    faq_count: int,
    doc_chunks: int,
) -> dict[str, Any]:
    """生成入库质量报告，并附带本次实际写入统计。

    调用顺序：命令行入口 -> build_quality_report()。
    """
    report = build_ingestion_quality_report(
        scenario_id=scenario.scenario_id,
        data_dir=args.data_dir or scenario.data_root,
        faq_csv=args.faq_csv or scenario.faq_csv_path,
        kb_version=kb_version,
        tenant_id=args.tenant_id,
        dataset_id=args.dataset_id,
        visibility=args.visibility,
        allowed_roles=args.allowed_role,
    )
    report["actual_ingest"] = {
        "faq_records_written": faq_count,
        "doc_chunks_written": doc_chunks,
        "activated": False,
    }
    return report


def enforce_quality_gate_or_exit(args: argparse.Namespace, report: dict[str, Any], kb_version: str) -> None:
    """执行质量门禁；激活路径默认执行，失败时保存报告并以非零状态退出。

    调用顺序：命令行入口 -> enforce_quality_gate_or_exit()。
    """
    if not args.quality_gate:
        return

    gate_result = evaluate_report_against_gate(report, quality_thresholds_from_args(args))
    report["quality_gate"] = gate_result
    if gate_result["ok"]:
        return

    report_path = save_ingestion_quality_report(report)
    print(
        "Ingestion quality gate failed; knowledge base version was not activated: "
        f"{kb_version}, quality_report={report_path}"
    )
    sys.exit(1)


def finalize_version(
    args: argparse.Namespace,
    scenario: Any,
    version_store: Any,
    kb_version: str,
    *,
    faq_count: int,
    doc_chunks: int,
) -> tuple[str, bool]:
    """质量检查和版本激活收口。

    调用顺序：命令行入口 -> finalize_version()。
    """
    if args.skip_quality_report:
        return "", False

    report = build_quality_report(args, scenario, kb_version, faq_count=faq_count, doc_chunks=doc_chunks)
    enforce_quality_gate_or_exit(args, report, kb_version)

    activated = False
    if args.activate:
        version_store.activate_version(kb_version)
        activated = True
        report["actual_ingest"]["activated"] = True

    return save_ingestion_quality_report(report), activated


def print_summary(
    *,
    kb_version: str,
    faq_count: int,
    doc_chunks: int,
    activated: bool,
    incremental_base_kb_version: str | None,
    report_path: str,
) -> None:
    """输出本次构建结果。

    调用顺序：命令行入口 -> print_summary()。
    """
    print(
        "Rebuilt knowledge base version: "
        f"{kb_version}, faq_records={faq_count}, doc_chunks={doc_chunks}, "
        f"activated={activated}, incremental_base={incremental_base_kb_version or 'none'}, "
        f"quality_report={report_path or 'skipped'}"
    )


def main() -> None:
    """按固定发布流程构建一个完整知识库版本。

    流程：解析场景 → 重置Collection(可选) → 创建目标版本 → 解析增量基准 →
         FAQ入库 → 文档入库 → 质量门禁 → 激活版本 → 输出摘要

    调用顺序：命令行入口 -> main()。
    """
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    # 第一步：解析场景配置并初始化 MySQL schema
    scenario = resolve_scenario(args.scenario)
    bootstrap_mysql_schema()

    # 第二步：按需重置 Milvus Collection（schema 变更时使用）
    reset_collections_if_requested(args, scenario)

    # 第三步：创建或复用目标知识库版本（STAGED 状态，不影响线上查询）
    version_store, version = ensure_target_version(args, scenario)
    kb_version = version.kb_version

    # 第四步：解析增量构建基准版本
    # 增量模式下，未变化文件的 chunk 直接引用旧版本，不重新 embedding
    incremental_base_kb_version = resolve_incremental_base(parser, args, version_store, kb_version)

    # 第五步：FAQ 入库（FAQ 不做增量，每次全量重建）
    faq_count = ingest_faq(args, scenario, kb_version)

    # 第六步：文档入库（支持增量：按文件 fingerprint 判断是否需要重新处理）
    doc_chunks = ingest_documents(args, scenario, kb_version, incremental_base_kb_version)

    # 第七步：质量报告 + 质量门禁 + 激活版本
    report_path, activated = finalize_version(
        args, scenario, version_store, kb_version,
        faq_count=faq_count, doc_chunks=doc_chunks,
    )

    # 第八步：输出构建摘要
    print_summary(
        kb_version=kb_version, faq_count=faq_count, doc_chunks=doc_chunks,
        activated=activated, incremental_base_kb_version=incremental_base_kb_version,
        report_path=report_path,
    )


if __name__ == "__main__":
    main()
