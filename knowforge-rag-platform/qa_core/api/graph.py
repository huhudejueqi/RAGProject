"""知识图谱查询 API 路由。

提供三个端点：
  GET  /api/graph/query?q=xxx     — 按实体名搜索，返回图数据
  GET  /api/graph/entity/:name    — 查看单实体上下文
  GET  /graph                     — 知识图谱可视化页面
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Query
from fastapi.responses import FileResponse

from qa_core.config.logging_config import get_logger
from qa_core.knowledge_graph.storage import GraphStorage

router = APIRouter()
logger = get_logger(__name__)
_storage = GraphStorage()


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
        entities = client.query(
            collection_name=_storage._entity_collection,
            filter=f'name like "%{q}%"',
            limit=top_k,
            output_fields=["name", "type", "description"],
        )

        # 2. 找这些实体的关系
        entity_names = [e["name"] for e in entities]
        all_relations = []
        related_names = set(entity_names)

        for name in entity_names:
            rels = client.query(
                collection_name=_storage._relation_collection,
                filter=f'source == "{name}" or target == "{name}"',
                limit=50,
                output_fields=["source", "target", "description", "strength"],
            )
            for r in rels:
                all_relations.append(r)
                related_names.add(r["source"])
                related_names.add(r["target"])

        # 3. 补上相关但不在初始结果里的实体
        extra_names = related_names - set(entity_names)
        if extra_names:
            for ename in extra_names:
                extra = client.query(
                    collection_name=_storage._entity_collection,
                    filter=f'name == "{ename}"',
                    limit=1,
                    output_fields=["name", "type", "description"],
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
                "label": r.get("description", "")[:20],
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

        entities = client.query(
            collection_name=_storage._entity_collection,
            filter=f'name == "{name}"',
            limit=1,
            output_fields=["name", "type", "description"],
        )
        if not entities:
            return {"error": "实体不存在"}

        rels = client.query(
            collection_name=_storage._relation_collection,
            filter=f'source == "{name}" or target == "{name}"',
            limit=50,
            output_fields=["source", "target", "description", "strength"],
        )

        # 邻居实体名
        neighbor_names = set()
        for r in rels:
            neighbor_names.add(r["source"])
            neighbor_names.add(r["target"])
        neighbor_names.discard(name)

        neighbors = []
        for nname in neighbor_names:
            n = client.query(
                collection_name=_storage._entity_collection,
                filter=f'name == "{nname}"',
                limit=1,
                output_fields=["name", "type", "description"],
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
