"""知识图谱社区摘要生成与社区级全局搜索。

社区搜索面向“不指向某个具体实体”的全局/综合问题：
构建阶段为每个社区生成摘要，查询阶段按摘要相关性返回最匹配的社区报告。
"""

from __future__ import annotations

import logging
import re
from typing import Any

import networkx as nx

from qa_core.config.logging_config import get_logger
from qa_core.config.settings import get_settings
from qa_core.knowledge_graph.graph_builder import Community
from qa_core.knowledge_graph.prompts import COMMUNITY_SUMMARY_PROMPT
from qa_core.knowledge_graph.storage import GraphStorage

logger = get_logger(__name__)

_MAX_ENTITY_LINES = 60
_MAX_RELATION_LINES = 120
_MAX_DESC_CHARS = 200
_MAX_SUMMARY_CHARS = 1000
_MAX_CONTEXT_CHARS = 12000


class CommunitySummarizer:
    """使用 LLM 为每个检测到的社区生成摘要。"""

    def __init__(self, llm: Any = None):
        self._llm = llm

    def _get_llm(self):
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            settings = get_settings()
            self._llm = ChatOpenAI(
                model=settings.llm_model,
                api_key=settings.llm_api_key,
                base_url=settings.llm_base_url,
                temperature=settings.llm_temperature,
                timeout=settings.llm_timeout,
                streaming=False,
            )
        return self._llm

    async def summarize_many(
        self,
        graph: nx.Graph,
        communities: list[Community],
    ) -> list[Community]:
        """逐社区生成摘要；单个社区失败不阻塞图入库。"""
        for community in communities:
            try:
                summary = await self.summarize(graph, community)
                community.summary = summary
                logger.info(
                    "社区 %d 摘要生成完成：%d 字",
                    community.community_id,
                    len(summary),
                )
            except Exception as e:
                logger.warning("社区 %d 摘要生成失败: %s", community.community_id, e)
        return communities

    async def summarize(self, graph: nx.Graph, community: Community) -> str:
        """为单个社区生成中文摘要。"""
        entity_text, relation_text = build_community_context(graph, community)
        if not entity_text.strip() and not relation_text.strip():
            return ""

        prompt = COMMUNITY_SUMMARY_PROMPT.format(
            community_entities=entity_text,
            community_relations=relation_text,
        )
        from langchain_core.messages import HumanMessage

        llm = self._get_llm()
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = str(getattr(response, "content", "") or "")
        return content.strip()[:_MAX_SUMMARY_CHARS]


def build_community_context(
    graph: nx.Graph,
    community: Community,
    max_entities: int = _MAX_ENTITY_LINES,
    max_relations: int = _MAX_RELATION_LINES,
    max_context_chars: int = _MAX_CONTEXT_CHARS,
) -> tuple[str, str]:
    """把社区包含的实体和关系格式化为摘要 Prompt 输入。

    max_context_chars 是实体/关系行的总字符预算，防止大社区把过长描述一次性塞给 LLM。
    """
    entity_names = list(community.entities)[:max_entities]
    entity_lines = []
    budget = max_context_chars
    for name in entity_names:
        data = graph.nodes.get(name, {})
        desc = str(data.get("description", "") or "")[:_MAX_DESC_CHARS]
        line = f"- {name}（{data.get('type', '其他')}）：{desc or '无描述'}"
        if budget - len(line) < 0:
            break
        budget -= len(line)
        entity_lines.append(line)

    relation_lines = []
    subgraph = graph.subgraph(entity_names)
    for u, v, data in subgraph.edges(data=True):
        if len(relation_lines) >= max_relations or budget <= 0:
            break
        label = data.get("label", "")
        desc = str(data.get("description", "") or "")[:_MAX_DESC_CHARS]
        line = f"- {u} → {v}（{label or '关联'}）：{desc or '无描述'}"
        if budget - len(line) < 0:
            break
        budget -= len(line)
        relation_lines.append(line)

    return "\n".join(entity_lines), "\n".join(relation_lines)


def _text_terms(text: str) -> set[str]:
    """提取查询/文本中的中文连续词和英文 token。"""
    terms: set[str] = set()
    for token in re.findall(r"[\u4e00-\u9fffA-Za-z0-9_]+", text or ""):
        if not token:
            continue
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            for start in range(len(token)):
                for end in range(start + 2, min(start + 7, len(token) + 1)):
                    terms.add(token[start:end])
        else:
            terms.add(token.lower())
    return terms


def score_communities(
    query: str,
    communities: list[dict[str, Any]],
) -> list[tuple[float, dict[str, Any]]]:
    """按社区摘要和实体与查询的文本重合度打分。"""
    query_terms = _text_terms(query)
    scored: list[tuple[float, dict[str, Any]]] = []
    for community in communities:
        summary_terms = _text_terms(community.get("summary", ""))
        entity_terms = _text_terms(community.get("entities", ""))
        score = 0.0
        score += len(query_terms & summary_terms) * 2.0
        score += len(query_terms & entity_terms) * 0.5
        if score > 0:
            scored.append((score, community))
    scored.sort(key=lambda item: (-item[0], item[1].get("community_id", 0)))
    return scored


def global_search(
    query: str,
    storage: GraphStorage | None = None,
    max_communities: int = 5,
    max_results: int = 200,
) -> str:
    """检索社区摘要并格式化为全局知识上下文。"""
    if not query or not query.strip():
        return ""

    try:
        active_storage = storage or GraphStorage()
        communities = active_storage.search_communities(max_results=max_results)
        communities = [
            c for c in communities if str(c.get("summary", "") or "").strip()
        ]
        if not communities:
            logger.debug("global_search: no community summaries, query=%s", query)
            return ""

        scored = score_communities(query, communities)
        if not scored:
            logger.debug("global_search: no relevant community, query=%s", query)
            return ""

        top = scored[:max_communities]
        lines = ["\n[社区知识参考]", f"社区范围: {len(top)} 个社区"]
        for score, community in top:
            summary = str(community.get("summary", "") or "")[:_MAX_SUMMARY_CHARS]
            lines.append(
                f"社区 {community.get('community_id', '?')}（相关度 {score:.1f}）：{summary}"
            )
        text = "\n".join(lines)
        logger.info(
            "global_search done: query=%s communities=%d",
            query, len(top),
        )
        return text
    except Exception as e:
        logger.warning("global_search failed: %s", e)
        return ""
