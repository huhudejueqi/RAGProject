"""轻量知识图谱可视化服务器。

独立于主 RAG 服务启动，不需要 MySQL/模型依赖。
搜索时做 2 跳展开 + 实体/边去重。
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
import uvicorn

from qa_core.knowledge_graph.storage import GraphStorage

app = FastAPI(title="KnowForge 知识图谱")
storage = GraphStorage(collection_name_prefix=os.environ.get("KG_PREFIX", ""))

static_dir = Path(__file__).resolve().parent / "static"


@app.get("/")
@app.get("/graph")
async def graph_page():
    return FileResponse(str(static_dir / "graph.html"),
                        headers={"Cache-Control": "no-store"})


def _dedup_entities(raw: list[dict]) -> list[dict]:
    """按 name 去重，保留第一个出现的 type/description。"""
    seen: dict[str, dict] = {}
    for e in raw:
        name = e.get("name", "")
        if name not in seen:
            seen[name] = e
    return list(seen.values())


def _dedup_edges(raw: list[dict]) -> list[dict]:
    """按 (source, target) 有序对去重，合并 description。"""
    seen: dict[tuple[str, str], dict] = {}
    for r in raw:
        s, t = r.get("source", ""), r.get("target", "")
        key = (s, t) if s < t else (t, s)
        if key in seen:
            old = seen[key]
            old_desc = old.get("description", "")
            new_desc = r.get("description", "")
            if new_desc and new_desc not in old_desc:
                old["description"] = old_desc + " | " + new_desc
            old["strength"] = max(old.get("strength", 1), r.get("strength", 1))
        else:
            seen[key] = dict(r)
    return list(seen.values())


@app.get("/api/graph/query")
async def query_graph(q: str = Query(""), top_k: int = Query(15)):
    if not q.strip():
        return {"nodes": [], "edges": []}
    try:
        client = storage._get_client()
        ecol = storage._entity_collection
        rcol = storage._relation_collection

        # ── 1. 搜实体（模糊匹配 name） ──
        raw_entities = client.query(
            collection_name=ecol,
            filter=f'name like "%{q}%"',
            limit=top_k,
            output_fields=["name", "type", "description"],
        )
        entities = _dedup_entities(raw_entities)

        # ── 2. 一阶关系 + 邻居 ──
        entity_names = {e["name"] for e in entities}
        all_relations: list[dict] = []
        neighbor_names: set[str] = set()

        for name in entity_names:
            rels = client.query(
                collection_name=rcol,
                filter=f'source == "{name}" or target == "{name}"',
                limit=80,
                output_fields=["source", "target", "label", "description", "strength"],
            )
            all_relations.extend(rels)
            for r in rels:
                neighbor_names.add(r["source"])
                neighbor_names.add(r["target"])

        # ── 3. 补邻居实体 ──
        missing = neighbor_names - entity_names
        for m in missing:
            hit = client.query(
                collection_name=ecol,
                filter=f'name == "{m}"',
                limit=1,
                output_fields=["name", "type", "description"],
            )
            if hit:
                entities.append(hit[0])

        # ── 4. 二跳展开（对一阶邻居再查关系） ──
        # 上一轮所有已知节点（含一阶邻居）
        known_nodes = entity_names | missing
        for neighbor in missing:
            hop2_rels = client.query(
                collection_name=rcol,
                filter=f'source == "{neighbor}" or target == "{neighbor}"',
                limit=40,
                output_fields=["source", "target", "label", "description", "strength"],
            )
            for r in hop2_rels:
                s, t = r["source"], r["target"]
                # 只保留至少一端在 already-known 节点中
                if s in known_nodes or t in known_nodes:
                    all_relations.append(r)
                    for hop_name in (s, t):
                        if hop_name not in known_nodes:
                            known_nodes.add(hop_name)
                            hit = client.query(
                                collection_name=ecol,
                                filter=f'name == "{hop_name}"',
                                limit=1,
                                output_fields=["name", "type", "description"],
                            )
                            if hit:
                                entities.append(hit[0])

        # ── 5. 最终去重 ──
        entities = _dedup_entities(entities)
        all_relations = _dedup_edges(all_relations)

        # ── 6. 构建 vis.js 格式 ──
        TYPE_COLORS = {
            "人物": "#3498db", "组织": "#e74c3c", "地点": "#2ecc71",
            "事件": "#f39c12", "概念": "#9b59b6", "项目": "#1abc9c",
            "产品": "#e67e22", "文档": "#34495e", "药品": "#27ae60",
            "疾病": "#c0392b", "症状": "#e67e22",
            "斗技": "#e67e22", "功法": "#8e44ad", "异火": "#e74c3c",
        }

        # 计算节点度
        known_set = {e["name"] for e in entities}
        degree: dict[str, int] = {}
        for r in all_relations:
            if r["source"] in known_set and r["target"] in known_set:
                degree[r["source"]] = degree.get(r["source"], 0) + 1
                degree[r["target"]] = degree.get(r["target"], 0) + 1
        max_deg = max(degree.values()) if degree else 1

        nodes = [
            {
                "id": e["name"],
                "label": e["name"],
                "title": f"[{e.get('type','?')}] {e.get('description','')[:120]}",
                "group": e.get("type", "OTHER"),
                "color": TYPE_COLORS.get(e.get("type", ""), "#95a5a6"),
                "degree": degree.get(e["name"], 0),
                "size": max(10, min(50, 10 + degree.get(e["name"], 0) / max_deg * 40)),
            }
            for e in entities
        ]

        edges = [
            {
                "from": r["source"],
                "to": r["target"],
                "label": r.get("label", "") or "",
                "title": (r.get("description", "") or "")[:120],
                "value": float(r.get("strength", 1)),
                "width": max(0.5, min(5, float(r.get("strength", 1)) / 2)),
            }
            for r in all_relations
            if r["source"] in known_set and r["target"] in known_set
        ]

        return {"nodes": nodes, "edges": edges, "stats": {
            "nodes": len(nodes), "edges": len(edges), "query": q,
        }}

    except Exception as e:
        import traceback
        return {"nodes": [], "edges": [], "error": str(e),
                "traceback": traceback.format_exc()}


if __name__ == "__main__":
    port = int(os.environ.get("KG_PORT", 18080))
    print(f"🐈 知识图谱服务: http://127.0.0.1:{port}/graph")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
