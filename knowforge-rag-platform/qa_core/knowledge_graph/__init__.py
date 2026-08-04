"""知识图谱构建与检索模块。

基于 GraphRAG 思想，提供从文档中提取实体和关系、构建知识图谱、
社群检测、图增强检索的完整链路。

核心流程：
  1. extractor.py        — LLM 驱动的实体与关系抽取
  2. graph_builder.py    — NetworkX 图构建与社群检测
  3. storage.py          — 实体/关系/社群写入 Milvus
  4. pipeline.py         — 与现有索引管线的集成入口
  5. local_search.py     — GraphRAG 本地搜索（基于图遍历）
  6. retrieval_integration.py — 与 RAG 管线的集成
  7. community_search.py — 社区摘要生成与社区级全局搜索
"""

from qa_core.knowledge_graph.extractor import GraphExtractor
from qa_core.knowledge_graph.graph_builder import KnowledgeGraphBuilder
from qa_core.knowledge_graph.storage import GraphStorage
from qa_core.knowledge_graph.pipeline import (
    run_knowledge_graph_pipeline,
    KGIngestResult,
)
from qa_core.knowledge_graph.local_search import local_search
from qa_core.knowledge_graph.community_search import global_search

__all__ = [
    "GraphExtractor",
    "KnowledgeGraphBuilder",
    "GraphStorage",
    "run_knowledge_graph_pipeline",
    "KGIngestResult",
    "local_search",
    "global_search",
]
