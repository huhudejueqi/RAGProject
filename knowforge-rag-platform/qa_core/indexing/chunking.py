"""文档切分策略。把标准化后的 Document 切成适合 Milvus 检索的父子块。"""

from __future__ import annotations
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from qa_core.config.settings import get_settings
from qa_core.document_metadata import is_reviewed_ocr_metadata, is_table_metadata
from qa_core.utils import stable_hash

CHINESE_SEPARATORS = [
    "\n\n",
    "\n",
    "。", "！", "？", "；",
    ";", ".", "!", "?",
    "，", ",",
    " ",
    "",
    # 原因： 中英文混排文档需要同时支持中文句号/感叹号/问号和英文句点/分号作为切分边界，递归切分器按 separator 顺序优先匹配大粒度分隔符
]

def chunk_identity(page_content: str, metadata: dict) -> tuple[str, str]:
    """基于正文和标准元数据生成 parent_id 与 chunk_id。

    调用顺序：入库脚本或索引服务 -> chunk_identity()。
    """
    parent_content = str(metadata.get("parent_content") or page_content or "").strip()
    if is_table_metadata(metadata):
        # 表格行的 parent_id 额外包含 table_id、sheet_name、row_number，确保同一张表的同一行
        # 在多次入库时 id 稳定，且不同行的 chunk 不会混淆；行列关系由 table_id + row_number 唯一锁定
        parent_id = stable_hash(
            metadata.get("scenario_id"),
            metadata.get("kb_version"),
            metadata.get("embedding_model_version"),
            metadata.get("chunk_schema_version"),
            metadata.get("doc_id"),
            metadata.get("table_id"),
            metadata.get("sheet_name"),
            metadata.get("row_number"),
            parent_content,
        )
        # 表格行的子块 id 只基于 parent_id + parent_content 生成，因为表格行不会进一步切分子块
        chunk_id = stable_hash(parent_id, parent_content)
        return parent_id, chunk_id

    parent_id = stable_hash(
        metadata.get("scenario_id"),
        metadata.get("kb_version"),
        metadata.get("embedding_model_version"),
        metadata.get("chunk_schema_version"),
        metadata.get("doc_id"),
        parent_content,
    )
    # 普通文档的 chunk_id 使用子块自身的 page_content（而非 parent_content）参与 hash，
    # 使得同一父块下不同子块拥有不同 chunk_id
    chunk_id = stable_hash(parent_id, page_content)
    return parent_id, chunk_id


def split_documents(documents: list[Document]) -> tuple[list[Document], list[str]]:
    """将文档切成可检索的子块并保留父块上下文。子块用于精确召回，parent_content 保存在 metadata 中。

    调用顺序：入库脚本或索引服务 -> split_documents()。
    Returns (chunks_list, ids_list)."""
    # 原因： parent-child 分别切分使子块保持精确命中而父块提供完整上下文窗口，比单一切片在精确召回率和上下文完整性之间取得更好平衡
    settings = get_settings()
    markdown_headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.parent_chunk_size,
        chunk_overlap=settings.parent_overlap,
        separators=CHINESE_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.child_chunk_size,
        chunk_overlap=settings.child_overlap,
        separators=CHINESE_SEPARATORS,
    )

    chunks: list[Document] = []
    ids: list[str] = []
    for doc in documents:
        file_type = str(doc.metadata.get("file_type", "")).lower()
        parent_docs: list[Document]
        if is_table_metadata(doc.metadata) or is_reviewed_ocr_metadata(doc.metadata):
            # 表格行和已复核 OCR 文本都是治理后的完整证据单元。
            # 表格不能被拆散行列关系；OCR 复核稿不能丢失复核状态、置信度和原始文件说明。
            parent_content = doc.page_content.strip()
            # 空行跳过：表格中全空行或 OCR 空白页没有检索价值，不生成 chunk 以节省 Milvus 存储
            if not parent_content:
                continue
            metadata = dict(doc.metadata)
            metadata["parent_content"] = parent_content
            parent_id, chunk_id = chunk_identity(parent_content, metadata)
            metadata.update(
                {
                    "parent_id": parent_id,
                    "chunk_id": chunk_id,
                }
            )
            # 表格/OCR 不经过父子切分：直接以整行/整段作为唯一块，parent_id == chunk_id
            chunks.append(Document(page_content=parent_content, metadata=metadata))
            ids.append(chunk_id)
            continue
        elif file_type == ".md":
            header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=markdown_headers)
            # Markdown 标题会先转成结构化元数据，再进入递归切分，能提升来源标签和上下文质量。
            # 如果 Markdown 解析失败，说明资料格式需要修复；入库阶段应该暴露异常并进入
            # 异常文件报告，而不是悄悄按普通文本切分，造成章节 metadata 丢失。
            header_docs = header_splitter.split_text(doc.page_content)
            for header_doc in header_docs:
                # Markdown 标题切分器会生成新的 Document，这里把原始文件 metadata
                # 补回去，避免切分后丢失 source、file_name、doc_id 等关键字段。
                header_doc.metadata.update(doc.metadata)
            parent_docs = parent_splitter.split_documents(header_docs)
        else:
            # 非 Markdown 普通文本：直接按父块大小切分，不再经过标题解析
            parent_docs = parent_splitter.split_documents([doc])

        for parent_doc in parent_docs:
            parent_content = parent_doc.page_content
            # parent_id 和 chunk_id 都纳入 kb_version、embedding_model_version 和 chunk_schema_version。
            # 这样同一个文件在两个知识库版本里可以同时存在，不会因为内容相同而主键冲突。
            child_docs = child_splitter.split_documents([parent_doc])
            # 子块为空意味着父块小于最小子块尺寸，此时跳过（父块本身仍可检索）
            for child_doc in child_docs:
                # chunk_id 由父块和子块内容共同决定。同一文件未变化时 id 稳定；文件变化时
                # id 会变化，配合 manifest 删除旧 chunk 后重建。
                metadata = dict(child_doc.metadata)
                metadata["parent_content"] = parent_content
                parent_id, chunk_id = chunk_identity(child_doc.page_content, metadata)
                metadata.update(
                    {
                        "parent_id": parent_id,
                        "chunk_id": chunk_id,
                    }
                )
                chunks.append(Document(page_content=child_doc.page_content, metadata=metadata))
                ids.append(chunk_id)
    return chunks, ids


