"""知识图谱查询 API 路由。

提供三个端点：
  GET  /api/graph/query?q=xxx     — 按实体名搜索，返回图数据
  GET  /api/graph/entity/:name    — 查看单实体上下文
  GET  /graph                     — 知识图谱可视化页面
"""

from __future__ import annotations

from pathlib import Path
import re

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from qa_core.config.logging_config import get_logger
from qa_core.knowledge_graph.storage import GraphStorage

router = APIRouter()
logger = get_logger(__name__)
_storage = GraphStorage()


def _graph_search_keys(query: str) -> list[str]:
    """把自然语言问题拆成可用的图谱实体搜索词。

    先保留完整问题，搜不到实体时再按常见提问词切出主体，例如
    “糖尿病怎么办”会追加“糖尿病”，“怎么治疗肺炎”会追加“肺炎”。
    """
    cleaned = re.sub(r"[\s\u3000，。？！、；：,.!?;:]+", "", query or "")
    if not cleaned:
        return []

    keys = [cleaned]
    markers = [
        "怎么办",
        "怎么治疗",
        "怎么调理",
        "怎么治",
        "如何治疗",
        "如何应对",
        "需要注意",
        "怎么",
        "如何",
        "有什么",
        "有哪些",
        "是什么",
        "应该",
        "可以",
    ]
    for marker in markers:
        if marker not in cleaned:
            continue
        before, _, after = cleaned.partition(marker)
        candidate = before.strip() if len(before.strip()) >= 2 else after.strip()
        if len(candidate) >= 2:
            keys.append(candidate)
        break
    return list(dict.fromkeys(keys))


def _static_page(path: str):
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@router.get("/graph")
async def graph_page():
    """知识图谱可视化页面。"""
    return _static_page(str(Path(__file__).resolve().parent.parent.parent / "static" / "graph.html"))


@router.get("/api/graph/query")
async def query_graph(
    q: str = Query("", description="搜索关键词"),
    top_k: int = Query(10, description="返回实体数"),
):
    """按名称模糊搜索实体，返回实体+关系+社群。"""
    if not q.strip():
        return {"nodes": [], "edges": [], "communities": []}

    try:
        client = _storage._get_client()

        # 1. 搜实体
        entities: list[dict] = []
        for search_key in _graph_search_keys(q):
            entities = list(
                client.query(
                    collection_name=_storage._entity_collection,
                    filter=f'name like "%{search_key}%"',
                    limit=top_k,
                    output_fields=["name", "type", "description"],
                )
            )
            if entities:
                break
        if not entities:
            return {"nodes": [], "edges": []}

        # 2. 找这些实体的关系
        entity_names = [e["name"] for e in entities]
        all_relations = []
        related_names = set(entity_names)

        for name in entity_names:
            rels = client.query(
                collection_name=_storage._relation_collection,
                filter=f'source == "{name}" or target == "{name}"',
                limit=50,
                output_fields=["source", "target", "label", "description", "strength"],
            )
            for r in rels:
                all_relations.append(r)
                related_names.add(r["source"])
                related_names.add(r["target"])

        # 3. 补上相关但不在初始结果里的实体
        extra_names = related_names - set(entity_names)
        if extra_names:
            for ename in extra_names:
                extra = list(
                    client.query(
                        collection_name=_storage._entity_collection,
                        filter=f'name == "{ename}"',
                        limit=1,
                        output_fields=["name", "type", "description"],
                    )
                )
                entities.extend(extra)

        # 4. 构建 vis-network 数据
        nodes = [
            {
                "id": e["name"],
                "label": e["name"],
                "title": f"{e.get('type','?')}: {e.get('description','')[:80]}",
                "group": e.get("type", "OTHER"),
            }
            for e in entities
        ]

        edges = [
            {
                "from": r["source"],
                "to": r["target"],
                "label": (r.get("label") or r.get("description", ""))[:20],
                "title": r.get("description", ""),
                "value": float(r.get("strength", 1)),
                "color": {"color": "#666", "opacity": 0.6},
            }
            for r in all_relations
            if r["source"] in related_names and r["target"] in related_names
        ]

        return {"nodes": nodes, "edges": edges}

    except Exception as e:
        logger.error("图查询失败: %s", e)
        return {"nodes": [], "edges": [], "error": str(e)}


@router.get("/api/graph/entity/{name}")
async def entity_detail(name: str):
    """查看单个实体的图上下文。"""
    try:
        client = _storage._get_client()

        entities = list(
            client.query(
                collection_name=_storage._entity_collection,
                filter=f'name == "{name}"',
                limit=1,
                output_fields=["name", "type", "description"],
            )
        )
        if not entities:
            return {"error": "实体不存在"}

        rels = client.query(
            collection_name=_storage._relation_collection,
            filter=f'source == "{name}" or target == "{name}"',
            limit=50,
            output_fields=["source", "target", "label", "description", "strength"],
        )

        # 邻居实体名
        neighbor_names = set()
        for r in rels:
            neighbor_names.add(r["source"])
            neighbor_names.add(r["target"])
        neighbor_names.discard(name)

        neighbors = []
        for nname in neighbor_names:
            n = list(
                client.query(
                    collection_name=_storage._entity_collection,
                    filter=f'name == "{nname}"',
                    limit=1,
                    output_fields=["name", "type", "description"],
                )
            )
            if n:
                neighbors.append(n[0])

        return {
            "entity": entities[0],
            "neighbors": neighbors,
            "relations": rels,
        }
    except Exception as e:
        return {"error": str(e)}
