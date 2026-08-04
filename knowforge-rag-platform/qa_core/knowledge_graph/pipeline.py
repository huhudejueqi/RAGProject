"""知识图谱构建管线：与现有索引服务集成。

在文档索引完成后自动触发实体/关系抽取和图构建。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken
from langchain_core.documents import Document

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.knowledge_graph.extractor import GraphExtractor, ExtractedEntity, ExtractedRelation
from qa_core.knowledge_graph.community_search import CommunitySummarizer
from qa_core.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from qa_core.knowledge_graph.storage import GraphStorage, EMBEDDING_DIM

logger = get_logger(__name__)

GRAPHRAG_BATCH_TOKEN_SIZE = 4096
_GRAPHRAG_TOKEN_ENCODING = "cl100k_base"


@dataclass
class KGIngestResult:
    """单次知识图谱构建的统计与错误结果。

    由 run_knowledge_graph_pipeline() 返回，用于调用方判断本次构建是否正常完成，
    并查看抽取、图构建、社群检测和 Milvus 写入各阶段的结果。

    字段语义：
    - total_chunks 是输入 Document 总数，未去重；processed_chunks 是去重后实际参与抽取的 parent 数量。
    - entities_extracted / relationships_extracted 是别名合并前的 LLM 抽取数量；合并后的图规模以 stored 和社区结果为准。
    - stored 只在图构建成功并写入 Milvus 后才有值；抽取为空或图构建失败时保持空 dict。
    - success 只表示没有记录到 errors，不等于一定产生了可用图谱：可能成功返回但实体/关系为空。
    - errors 可能来自抽取、图构建或存储阶段；调用方应检查其内容决定是否告警或回滚。
    """
    total_chunks: int = 0
    processed_chunks: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    communities_detected: int = 0
    stored: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """是否没有任何记录到的错误；注意不代表本次构建一定产生了可用图谱。"""
        return len(self.errors) == 0


@dataclass
class _GraphExtractionBatch:
    """按 parent_content 顺序累计后的一次抽取批次。"""

    text: str
    chunks: list[Document]
    source_chunk_ids: list[str]


def _num_tokens(text: str) -> int:
    """按 cl100k_base 估算文本 token 数，和 GraphRAG 默认分批口径一致。"""
    return len(tiktoken.get_encoding(_GRAPHRAG_TOKEN_ENCODING).encode(text))


def _graph_text_for_chunk(chunk: Document) -> str:
    """优先使用 parent_content；测试/无父子元数据时回退到 page_content。"""
    metadata = getattr(chunk, "metadata", None) or {}
    parent_content = metadata.get("parent_content")
    if parent_content and str(parent_content).strip():
        return str(parent_content).strip()
    page_content = getattr(chunk, "page_content", "")
    return str(page_content or "").strip()


def _graph_source_key(chunk: Document, text: str) -> tuple[object, ...]:
    """生成图谱抽取去重键：优先 parent_id，其次 chunk_id，最后按文档与文本兜底。"""
    metadata = getattr(chunk, "metadata", None) or {}
    if metadata.get("parent_id"):
        return ("parent_id", str(metadata["parent_id"]))
    if metadata.get("chunk_id"):
        return ("chunk_id", str(metadata["chunk_id"]))
    return ("doc", str(metadata.get("doc_id", "")), text)


def _build_graph_batches(
    chunks: list[Document],
    max_tokens: int = GRAPHRAG_BATCH_TOKEN_SIZE,
) -> list[_GraphExtractionBatch]:
    """把 parent_content 去重并按原始顺序累计到 token 上限。

    同一 parent_id 的多个 child chunk 只抽取一次；单个 parent 超过上限时
    仍作为一个 batch 交给抽取，避免再次切割 parent_content。
    """
    seen: set[tuple[object, ...]] = set()
    unique_parents: list[Document] = []
    for chunk in chunks:
        text = _graph_text_for_chunk(chunk)
        if not text:
            continue
        key = _graph_source_key(chunk, text)
        if key in seen:
            continue
        seen.add(key)
        unique_parents.append(chunk)

    batches: list[_GraphExtractionBatch] = []
    current_chunks: list[Document] = []
    current_text = ""
    current_tokens = 0

    def flush() -> None:
        nonlocal current_chunks, current_text, current_tokens
        if not current_chunks:
            return
        batches.append(_GraphExtractionBatch(
            text=current_text,
            chunks=list(current_chunks),
            source_chunk_ids=[
                str((getattr(chunk, "metadata", None) or {}).get("chunk_id") or "")
                for chunk in current_chunks
            ],
        ))
        current_chunks = []
        current_text = ""
        current_tokens = 0

    for chunk in unique_parents:
        text = _graph_text_for_chunk(chunk)
        separator = "\n\n" if current_text else ""
        candidate_text = f"{current_text}{separator}{text}"
        candidate_tokens = _num_tokens(candidate_text)

        if current_chunks and candidate_tokens > max_tokens:
            flush()
            candidate_text = text
            candidate_tokens = _num_tokens(candidate_text)

        current_text = candidate_text
        current_tokens = candidate_tokens
        current_chunks.append(chunk)

    flush()
    return batches


def _resolve_entity_aliases(
    entities: list[ExtractedEntity],
    relationships: list[ExtractedRelation],
    enabled: bool = True,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """实体别名解析：合并同一实体的不同名称。

    检测规则（相似度 >= 0.65 时合并）：
    - 字符重排：米特尔拍卖场 ↔ 特米尔拍卖场
    - 后缀匹配：萧薰儿 ← 薰儿
    - 高 Jaccard 重叠：萧薰儿 ↔ 萧熏儿, 成人仪式 ↔ 成年仪式
    - 安全阀防止误合：纳兰桀 ≠ 纳兰肃, 七年之约 ≠ 三年之约

    返回 (merged_entities, merged_relationships)
    """
    if not enabled or len(entities) < 2:
        return entities, relationships

    def _sim(a: str, b: str) -> float:
        if a == b or not a or not b:
            return 1.0 if a == b else 0.0
        short, long_ = (a, b) if len(a) <= len(b) else (b, a)
        same_len = len(a) == len(b)
        set_a, set_b = set(a), set(b)
        common = len(set_a & set_b)
        jac = common / max(len(set_a | set_b), 0.01)

        # 规则0: 字符重排
        if sorted(short) == sorted(long_):
            return 0.85
        # 安全阀1: 同姓+同中字+异末字 → 不同人
        if len(a) == len(b) == 3 and a[:1] == b[:1]:
            if a[1] == b[1] and a[2] != b[2]:
                return 0.0
            if a[1] != b[1] and a[2] == b[2]:
                return 0.7
        # 安全阀2: 同尾缀+首字不同 → 不同事件
        if same_len and len(a) >= 4 and a[2:] == b[2:] and a[0] != b[0]:
            return 0.0
        # 安全阀3: 同尾缀+颜色首字 → 不同物品
        if same_len and len(a) >= 3 and a[1:] == b[1:] and a[0] in {'黑','红','绿','紫','黄','白','蓝','青'}:
            return 0.0
        # 安全阀4: 等级后缀不合并
        RANKED = {'斗者','斗师','大斗师','斗灵','斗王','斗皇','斗宗','斗尊','斗圣','斗帝','斗之气','斗之气旋'}
        if not same_len and short in RANKED:
            return 0.0
        # 规则A: 同长度、高Jaccard
        if same_len:
            if common >= len(a) - 1 and jac >= 0.5:
                return 0.7
            if jac >= 0.75:
                return 0.7
        # 规则B: 后缀匹配
        if not same_len and len(short) >= 2 and long_.endswith(short):
            return 0.85
        # 规则D: 高Jaccard + 非层级前缀
        if jac >= 0.6 and common >= 2:
            is_prefix = not same_len and long_.startswith(short)
            suffix_len = len(long_) - len(short)
            if is_prefix and 1 <= suffix_len <= 2:
                generic_suffixes = {'家','族','会','集','赛','处','场','堂','石','牌'}
                if any(s in long_[len(short):] for s in generic_suffixes):
                    return 0.7
            if not is_prefix:
                return 0.7
        return 0.0

    # 类型兼容判断
    def _compat(t1: str, t2: str) -> bool:
        return t1 == t2

    from collections import defaultdict

    # 按类型分组
    by_type = defaultdict(list)
    for e in entities:
        by_type[e.type].append(e.name)

    # 构建别名图
    alias_graph = defaultdict(set)
    for etype, names in by_type.items():
        names = list(set(names))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if _sim(names[i], names[j]) >= 0.65:
                    alias_graph[names[i]].add(names[j])
                    alias_graph[names[j]].add(names[i])

    # 找联通分量
    visited = set()
    groups = []
    for name in alias_graph:
        if name in visited:
            continue
        stack = [name]
        group = set()
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            group.add(n)
            for neighbor in alias_graph[n]:
                if neighbor not in visited:
                    stack.append(neighbor)
        if len(group) > 1:
            groups.append(group)

    if not groups:
        return entities, relationships

    # 选规范名 (优先选带姓氏的)
    FAMILY_NAMES = {'萧','纳','兰','米','特','加','列','奥','巴','帕'}
    merged_to_canonical = {}
    for group in groups:
        has_family = [n for n in group if n[0] in FAMILY_NAMES]
        canonical = max(has_family, key=len) if has_family else max(group, key=len)
        for name in group:
            merged_to_canonical[name] = canonical

    logger = get_logger(__name__)
    logger.info("别名解析：发现 %d 组合并", len(groups))
    for g in groups:
        canonical = merged_to_canonical.get(list(g)[0], list(g)[0])
        logger.info("  → %s: %s", canonical, sorted(g))

    # 合并实体
    merged = {}
    for e in entities:
        name = merged_to_canonical.get(e.name, e.name)
        if name in merged:
            exist = merged[name]
            if e.description and e.description not in exist.description:
                exist.description += f"；{e.description}"
        else:
            merged[name] = ExtractedEntity(
                name=name, type=e.type, description=e.description,
                source_chunk_id=e.source_chunk_id,
            )

    # 更新关系
    seen = set()
    merged_rels = []
    for r in relationships:
        src = merged_to_canonical.get(r.source, r.source)
        tgt = merged_to_canonical.get(r.target, r.target)
        if src == tgt:
            continue
        pair = (src, tgt)
        if pair in seen or (tgt, src) in seen:
            continue
        seen.add(pair)
        merged_rels.append(ExtractedRelation(
            source=src, target=tgt, label=r.label,
            description=r.description, strength=r.strength,
        ))

    logger.info("别名解析结果：%d 实体 → %d, %d 关系 → %d",
                len(entities), len(merged), len(relationships), len(merged_rels))
    return list(merged.values()), merged_rels


async def run_knowledge_graph_pipeline(
    chunks: list[Document],
    kb_version: str = "",
    collection_prefix: str = "",
    max_tokens: int = GRAPHRAG_BATCH_TOKEN_SIZE,
    max_gleanings: int = 1,
    enable_community_detection: bool = True,
    generate_community_summaries: bool = True,
) -> KGIngestResult:
    """从文档块中执行完整的知识图谱构建管线。

    流程：
        1. 按 parent_content 去重并按文档顺序累计到 token 上限
        2. 每个 batch 调用一次 LLM 实体/关系抽取
        3. 构建 NetworkX 图
        4. （可选）社群检测
        5. （可选）为社群生成摘要，失败不阻塞图入库
        6. 结果存入 Milvus

    参数：
        chunks: 文档块列表（来自索引管线的输出）
        kb_version: 当前知识库版本号
        collection_prefix: 集合名称前缀
        max_tokens: 每个抽取 batch 的文本 token 上限
        max_gleanings: 每轮抽取的迭代补充次数
        enable_community_detection: 是否执行社群检测
        generate_community_summaries: 是否生成社区摘要，默认开启；测试或成本敏感场景可关闭

    返回：
        KGIngestResult: 构建统计与错误结果，字段语义见 KGIngestResult 类注释。
    """
    if not chunks:
        logger.warning("无可处理的文档块")
        return KGIngestResult()

    result = KGIngestResult(total_chunks=len(chunks))

    # 初始化组件
    extractor = GraphExtractor(max_gleanings=max_gleanings)
    builder = KnowledgeGraphBuilder()
    storage = GraphStorage(collection_name_prefix=collection_prefix)
    summarizer = CommunitySummarizer()

    # 确保集合已创建
    try:
        settings = get_settings()
        storage.ensure_collections(dim=EMBEDDING_DIM)
    except Exception as e:
        logger.warning("创建 Milvus 集合失败（可能无 Milvus 服务）: %s", e)

    # 按 parent_content 分批抽取
    graph_batches = _build_graph_batches(chunks, max_tokens=max_tokens)
    if not graph_batches:
        logger.info("没有足够长的 parent_content，跳过知识图谱构建")
        return result

    all_entities = []
    all_relationships = []

    for batch_idx, batch in enumerate(graph_batches, start=1):
        try:
            logger.info(
                "知识图谱抽取批次 %d/%d: %d 个 parent, %d tokens",
                batch_idx, len(graph_batches), len(batch.chunks), _num_tokens(batch.text),
            )
            extraction = await extractor.extract(batch.text)
            source_chunk_id = ",".join(batch.source_chunk_ids)
            for e in extraction.entities:
                e.source_chunk_id = source_chunk_id
            for r in extraction.relationships:
                r.source_chunk_id = source_chunk_id
            all_entities.extend(extraction.entities)
            all_relationships.extend(extraction.relationships)
            result.processed_chunks += len(batch.chunks)
        except Exception as e:
            err_msg = f"Graph extraction batch {batch_idx} 抽取失败: {e}"
            logger.error(err_msg)
            result.errors.append(err_msg)

    result.entities_extracted = len(all_entities)
    result.relationships_extracted = len(all_relationships)

    if not all_entities and not all_relationships:
        logger.info("未抽取到任何实体或关系，跳过图构建")
        return result

    # 别名解析：合并同名实体
    all_entities, all_relationships = _resolve_entity_aliases(
        all_entities, all_relationships, enabled=True,
    )

    # 构建知识图谱
    try:
        graph_result = builder.build(all_entities, all_relationships)
        result.communities_detected = graph_result.community_count
    except Exception as e:
        err_msg = f"图构建失败: {e}"
        logger.error(err_msg)
        result.errors.append(err_msg)
        return result

    # 社区摘要：供全局/综合问题检索使用，失败不阻塞图入库
    if enable_community_detection and generate_community_summaries and graph_result.communities:
        try:
            graph_result.communities = await summarizer.summarize_many(
                graph_result.graph, graph_result.communities,
            )
        except Exception as e:
            logger.warning("社区摘要生成失败: %s", e)


    # 存入 Milvus
    try:
        stored = storage.store_graph(graph_result, kb_version=kb_version)
        result.stored = stored
    except Exception as e:
        err_msg = f"图存储失败: {e}"
        logger.warning(err_msg)
        result.errors.append(err_msg)

    logger.info(
        "知识图谱管线完成: %d/%d chunks 处理, "
        "%d 实体, %d 关系, %d 社群, 错误=%d",
        result.processed_chunks, result.total_chunks,
        result.entities_extracted, result.relationships_extracted,
        result.communities_detected, len(result.errors),
    )
    return result


async def run_kg_pipeline_for_document(
    content: str,
    doc_id: str = "",
    kb_version: str = "",
) -> KGIngestResult:
    """为单篇文档运行知识图谱构建（快速入口）。

    适合测试或手动触发。
    """
    doc = Document(
        page_content=content,
        metadata={"chunk_id": doc_id or "doc_0", "source": doc_id},
    )
    return await run_knowledge_graph_pipeline(
        chunks=[doc],
        kb_version=kb_version,
    )
