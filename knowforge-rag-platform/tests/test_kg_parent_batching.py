"""知识图谱 parent_content 分批抽取测试。"""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.documents import Document

from qa_core.knowledge_graph.extractor import ExtractedEntity, ExtractionResult
from qa_core.knowledge_graph.pipeline import _build_graph_batches, run_knowledge_graph_pipeline


def _doc(
    chunk_id: str,
    parent_id: str | None = None,
    parent_content: str | None = None,
    page_content: str | None = None,
) -> Document:
    metadata = {"chunk_id": chunk_id}
    if parent_id is not None:
        metadata["parent_id"] = parent_id
    if parent_content is not None:
        metadata["parent_content"] = parent_content
    return Document(page_content=page_content or "", metadata=metadata)


class GraphBatchBuilderTests(unittest.TestCase):
    """验证 parent_content 去重、顺序和 token 分批逻辑。"""

    def test_uses_parent_content_and_deduplicates_same_parent(self) -> None:
        parent_text = "这是一个足够长的父块内容，用于验证知识图谱抽取会直接使用 parent_content。"
        chunks = [
            _doc("child_1", parent_id="parent_1", parent_content=parent_text),
            _doc("child_2", parent_id="parent_1", parent_content=parent_text),
        ]

        batches = _build_graph_batches(chunks, max_tokens=4096)

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].text, parent_text)
        self.assertEqual(batches[0].source_chunk_ids, ["child_1"])

    def test_falls_back_to_page_content_when_parent_content_missing(self) -> None:
        page_text = "没有 parent_content 时回退到 page_content 进行知识图谱抽取。"
        chunks = [_doc("child_1", page_content=page_text)]

        batches = _build_graph_batches(chunks, max_tokens=4096)

        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0].text, page_text)

    def test_accumulates_by_token_limit_and_preserves_order(self) -> None:
        chunks = [
            _doc("c1", parent_id="p1", parent_content="一" * 8),
            _doc("c2", parent_id="p2", parent_content="二" * 8),
            _doc("c3", parent_id="p3", parent_content="三" * 8),
        ]

        batches = _build_graph_batches(chunks, max_tokens=20)

        self.assertEqual(len(batches), 2)
        self.assertIn("一", batches[0].text)
        self.assertIn("二", batches[0].text)
        self.assertNotIn("三", batches[0].text)
        self.assertIn("三", batches[1].text)
        self.assertEqual(batches[1].source_chunk_ids, ["c3"])


class KnowledgeGraphPipelineBatchTests(unittest.IsolatedAsyncioTestCase):
    """验证每个 token batch 只调用一次实体/关系抽取。"""

    @patch("qa_core.knowledge_graph.pipeline.GraphExtractor.extract", new_callable=AsyncMock)
    @patch("qa_core.knowledge_graph.pipeline.GraphStorage.store_graph", return_value={
        "entities": 2,
        "relationships": 0,
        "communities": 0,
    })
    @patch("qa_core.knowledge_graph.pipeline.GraphStorage.ensure_collections")
    async def test_one_extraction_call_per_batch(
        self,
        mock_ensure_collections,
        mock_store_graph,
        mock_extract,
    ) -> None:
        chunks = [
            _doc("c1", parent_id="p1", parent_content="一" * 8),
            _doc("c2", parent_id="p2", parent_content="二" * 8),
            _doc("c3", parent_id="p3", parent_content="三" * 8),
        ]
        mock_extract.side_effect = lambda text: ExtractionResult(
            entities=[ExtractedEntity(
                name=f"实体{len(text)}",
                type="ENTITY",
                description=text[:20],
            )],
        )

        result = await run_knowledge_graph_pipeline(
            chunks,
            kb_version="test_kg_v1",
            max_tokens=20,
            generate_community_summaries=False,
        )

        self.assertEqual(mock_extract.await_count, 2)
        self.assertEqual(result.processed_chunks, 3)
        self.assertEqual(result.entities_extracted, 2)
        self.assertEqual(result.errors, [])
