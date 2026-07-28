"""知识图谱与 RAG 检索的集成工具。

使用 GraphRAG 本地搜索（local_search）作为知识图谱的主检索入口，
替代原有的简单字符串匹配。当本地搜索无结果时回退到简单搜索。
"""

from __future__ import annotations

from qa_core.config.logging_config import get_logger
from qa_core.knowledge_graph.local_search import local_search

logger = get_logger(__name__)


def format_graph_context(query: str, max_entities: int = 8, max_relations: int = 20) -> str:
    """搜索知识图谱，返回格式化后的图上下文文本。

    使用 GraphRAG 本地搜索：
      1. 从查询中提取实体关键词（含同义词扩展）
      2. 在 KG 中匹配实体
      3. N 跳图遍历获取子图
      4. 按相关度排序关系
      5. 格式化为结构化上下文

    参数：
        query: 用户查询
        max_entities: 返回的最大实体数
        max_relations: 返回的最大关系数

    返回：
        格式化的知识图谱上下文（无匹配时返回空字符串）
    """
    kg_ctx = local_search(query, max_entities=max_entities, max_relations=max_relations, max_hops=2)
    if kg_ctx:
        logger.info("format_graph_context: using local_search for query=%s", query)
    else:
        logger.debug("format_graph_context: local_search returned empty for query=%s", query)
    return kg_ctx
