"""GraphRAG 本地搜索：以知识图谱为主检索索引。

参考 Microsoft GraphRAG 论文 "From Local to Global"，
将知识图谱作为一级检索索引，通过实体匹配 + 图遍历
获取结构化的上下文。

流程：
  1. 从查询中提取实体关键词
  2. 在 KG 中匹配实体（Milvus like + name 索引）
  3. 从匹配实体出发，沿关系进行 N 跳图遍历
  4. 收集子图（实体 + 关系 + 社群归属）
  5. 按相关度和图距离排序
  6. 格式化为 LLM 可消费的结构化上下文
"""

from __future__ import annotations

from typing import Any

from qa_core.config.logging_config import get_logger
from qa_core.knowledge_graph.storage import GraphStorage

logger = get_logger(__name__)
_storage = GraphStorage()

# 近义词映射（从 retrieval_integration 复用）
_SYNONYM_MAP: dict[str, list[str]] = {
    "头疼": ["头痛", "偏头痛"],
    "腰部": ["腰"],
    "肚疼": ["腹痛", "肚子痛"],
    "胃疼": ["胃痛", "腹痛"],
    "感冒": ["流行性感冒"],
    "发烧": ["发热"],
    "拉肚子": ["腹泻"],
    "睡不着": ["失眠"],
    "口腔溃疡": ["口疮"],
    "皮肤痒": ["瘙痒"],
    "流鼻涕": ["鼻塞", "鼻炎"],
}


def _expand_tokens(query: str, min_len: int = 2, max_len: int = 6) -> list[str]:
    """从查询中提取实体候选 token，含同义词扩展。"""
    tokens: list[str] = []
    i = 0
    while i < len(query):
        if "\u4e00" <= query[i] <= "\u9fff":
            j = i
            while j < len(query) and "\u4e00" <= query[j] <= "\u9fff":
                j += 1
            chunk = query[i:j]
            for start in range(len(chunk)):
                for end in range(
                    start + min_len, min(start + max_len + 1, len(chunk) + 1)
                ):
                    t = chunk[start:end]
                    tokens.append(t)
                    if t in _SYNONYM_MAP:
                        tokens.extend(_SYNONYM_MAP[t])
            i = j
        else:
            i += 1
    seen: set[str] = set()
    unique: list[str] = []
    for t in sorted(set(tokens), key=len, reverse=True):
        if t not in seen:
            seen.add(t)
            unique.append(t)
    return unique


def _match_entities(
    client: Any,
    collection: str,
    tokens: list[str],
    max_entities: int = 20,
) -> list[dict]:
    """在 KG 中匹配实体，去重后返回。"""
    seen: dict[str, dict] = {}
    for token in tokens:
        try:
            results = client.query(
                collection_name=collection,
                filter='name like "%{}%"'.format(token),
                limit=max_entities,
                output_fields=["name", "type", "description"],
            )
            for e in results:
                if e["name"] not in seen:
                    seen[e["name"]] = e
        except Exception:
            continue
        if len(seen) >= max_entities:
            break
    return list(seen.values())


def _traverse_from_entities(
    client: Any,
    relation_collection: str,
    entity_names: list[str],
    max_hops: int = 2,
    max_relations_per_entity: int = 30,
) -> tuple[list[dict], list[dict]]:
    """从匹配实体出发，沿关系图遍历。

    返回 (所有涉及的实体列表, 所有涉及的关系列表)。
    """
    visited_entities = set(entity_names)
    all_entities = list(entity_names)
    all_relations: list[dict] = []
    frontier = list(entity_names)

    for hop in range(max_hops):
        if not frontier:
            break
        next_frontier: list[str] = []
        for name in frontier:
            try:
                rels = client.query(
                    collection_name=relation_collection,
                    filter='source == "{}" or target == "{}"'.format(name, name),
                    limit=max_relations_per_entity,
                    output_fields=["source", "target", "description", "strength"],
                )
            except Exception:
                continue
            for r in rels:
                all_relations.append(r)
                neighbor = r["target"] if r["source"] == name else r["source"]
                if neighbor not in visited_entities:
                    visited_entities.add(neighbor)
                    next_frontier.append(neighbor)
                    all_entities.append(neighbor)
        frontier = next_frontier

    # 去重
    seen_r: set[str] = set()
    deduped_rels: list[dict] = []
    for r in all_relations:
        key = "{}->{}:{}".format(r["source"], r["target"], r.get("description", ""))
        if key not in seen_r:
            seen_r.add(key)
            deduped_rels.append(r)

    return list(visited_entities), deduped_rels


def _rank_relations(
    relations: list[dict],
    seed_entities: set[str],
    max_relations: int = 30,
) -> list[dict]:
    """按相关度排序关系。

    评分规则：
      - 种子实体间的关系 +3
      - 种子实体与邻居的关系 +1
      - strength 加权
    """
    scored: list[tuple[float, dict]] = []
    for r in relations:
        src_in = r["source"] in seed_entities
        tgt_in = r["target"] in seed_entities
        if src_in and tgt_in:
            score = 3.0
        elif src_in or tgt_in:
            score = 1.0
        else:
            score = 0.5
        score *= float(r.get("strength", 1.0))
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:max_relations]]


def local_search(
    query: str,
    max_entities: int = 15,
    max_relations: int = 30,
    max_hops: int = 2,
) -> str:
    """GraphRAG 本地搜索：以知识图谱为主索引检索上下文。

    参数：
        query: 用户查询
        max_entities: 返回的最大实体数
        max_relations: 返回的最大关系数
        max_hops: 图遍历的最大跳数

    返回：
        格式化的知识图谱上下文（无匹配时返回空字符串）
    """
    try:
        client = _storage._get_client()

        # 1. 提取实体 token
        tokens = _expand_tokens(query)
        if not tokens:
            logger.debug("local_search: no tokens from query=%s", query)
            return ""

        # 2. 匹配种子实体
        seed_entities = _match_entities(
            client, _storage._entity_collection, tokens, max_entities
        )
        if not seed_entities:
            logger.debug("local_search: no entity match, query=%s tokens=%s", query, tokens[:8])
            return ""

        seed_names = [e["name"] for e in seed_entities]
        logger.info(
            "local_search: matched %d entities for query=%s: %s",
            len(seed_names), query, seed_names[:6],
        )

        # 3. 图遍历（N 跳）
        all_entity_names, all_relations = _traverse_from_entities(
            client, _storage._relation_collection,
            seed_names, max_hops=max_hops,
        )

        # 4. 排序关系
        seed_set = set(seed_names)
        ranked_relations = _rank_relations(all_relations, seed_set, max_relations)

        # 5. 获取邻居实体的详细信息
        neighbor_names = set(all_entity_names) - set(seed_names)
        neighbor_entities: list[dict] = []
        for ename in list(neighbor_names)[:max_entities]:
            try:
                extra = client.query(
                    collection_name=_storage._entity_collection,
                    filter='name == "{}"'.format(ename),
                    limit=1,
                    output_fields=["name", "type", "description"],
                )
                if extra:
                    neighbor_entities.append(extra[0])
            except Exception:
                continue

        # 6. 格式化
        lines = ["\n[知识图谱参考]"]

        # 容器摘要
        lines.append(
            "子图范围: {} 实体, {} 关系, {} 跳遍历".format(
                len(all_entity_names), len(ranked_relations), max_hops,
            )
        )

        # 种子实体
        parts = []
        for e in seed_entities[:max_entities]:
            parts.append(
                "{} ({}) [{}]".format(
                    e["name"], e.get("type", "?"), e.get("description", "")[:40]
                )
            )
        if parts:
            lines.append("相关实体：{}".format("；".join(parts)))

        # 关系（按评分排序输出）
        if ranked_relations:
            lines.append("关系路径：")
            for r in ranked_relations[:max_relations]:
                desc = r.get("description", "")
                strength = float(r.get("strength", 1))
                star = "★" if strength >= 7 else "☆"
                lines.append(
                    "  {} → {} [{}] {}".format(
                        r["source"], r["target"], desc[:40], star
                    )
                )

        text = "\n".join(lines)
        logger.info(
            "local_search done: query=%s entities=%d relations=%d hops=%d",
            query, len(all_entity_names), len(ranked_relations), max_hops,
        )
        return text

    except Exception as e:
        logger.warning("local_search failed: %s", e)
        return ""
