"""知识图谱构建管线：与现有索引服务集成。

在文档索引完成后自动触发实体/关系抽取和图构建。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.knowledge_graph.extractor import GraphExtractor
from qa_core.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from qa_core.knowledge_graph.storage import GraphStorage, EMBEDDING_DIM

logger = get_logger(__name__)


@dataclass
class KGIngestResult:
    """单次图谱构建的统计结果。"""
    total_chunks: int = 0
    processed_chunks: int = 0
    entities_extracted: int = 0
    relationships_extracted: int = 0
    communities_detected: int = 0
    stored: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return len(self.errors) == 0


async def run_knowledge_graph_pipeline(
    chunks: list[Document],
    kb_version: str = "",
    collection_prefix: str = "",
    batch_size: int = 10,
    max_gleanings: int = 1,
    enable_community_detection: bool = True,
) -> KGIngestResult:
    """从文档块中执行完整的知识图谱构建管线。

    流程：
        1. 对每个 chunk 执行 LLM 实体/关系抽取
        2. 构建 NetworkX 图
        3. （可选）社群检测
        4. 结果存入 Milvus

    参数：
        chunks: 文档块列表（来自索引管线的输出）
        kb_version: 当前知识库版本号
        collection_prefix: 集合名称前缀
        batch_size: 每批处理的 chunk 数
        max_gleanings: 每轮抽取的迭代补充次数
        enable_community_detection: 是否执行社群检测

    返回：
        KGIngestResult: 构建统计信息
    """
    if not chunks:
        logger.warning("无可处理的文档块")
        return KGIngestResult()

    result = KGIngestResult(total_chunks=len(chunks))

    # 初始化组件
    extractor = GraphExtractor(max_gleanings=max_gleanings)
    builder = KnowledgeGraphBuilder()
    storage = GraphStorage(collection_name_prefix=collection_prefix)

    # 确保集合已创建
    try:
        settings = get_settings()
        storage.ensure_collections(dim=EMBEDDING_DIM)
    except Exception as e:
        logger.warning("创建 Milvus 集合失败（可能无 Milvus 服务）: %s", e)

    # 分批抽取
    all_entities = []
    all_relationships = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for chunk in batch:
            chunk_text = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            chunk_id = getattr(chunk, "metadata", {}).get("chunk_id", str(i))

            if not chunk_text or len(chunk_text.strip()) < 20:
                continue  # 跳过过短的文本

            try:
                extraction = await extractor.extract(chunk_text)
                for e in extraction.entities:
                    e.source_chunk_id = chunk_id
                for r in extraction.relationships:
                    r.source_chunk_id = chunk_id
                all_entities.extend(extraction.entities)
                all_relationships.extend(extraction.relationships)
                result.processed_chunks += 1
            except Exception as e:
                err_msg = f"Chunk {chunk_id} 抽取失败: {e}"
                logger.error(err_msg)
                result.errors.append(err_msg)

    result.entities_extracted = len(all_entities)
    result.relationships_extracted = len(all_relationships)

    if not all_entities and not all_relationships:
        logger.info("未抽取到任何实体或关系，跳过图构建")
        return result

    # 构建知识图谱
    try:
        graph_result = builder.build(all_entities, all_relationships)
        result.communities_detected = graph_result.community_count
    except Exception as e:
        err_msg = f"图构建失败: {e}"
        logger.error(err_msg)
        result.errors.append(err_msg)
        return result

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
