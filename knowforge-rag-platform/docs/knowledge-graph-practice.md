# 知识图谱构建实践总结

## 目录

- [管线架构](#管线架构)
- [实体关系抽取](#实体关系抽取)
- [短标签生成](#短标签生成)
- [别名解析（实体对齐）](#别名解析实体对齐)
- [Milvus 存储设计](#milvus-存储设计)
- [可视化服务](#可视化服务)
- [与 Microsoft GraphRAG 对比](#与-microsoft-graphrag-对比)

---

## 管线架构

整个知识图谱管线串联在 `qa_core/knowledge_graph/pipeline.py` 的 `run_knowledge_graph_pipeline()` 中：

```
文档块 (list[Document])
  │
  ▼
GraphExtractor.extract()
  ├── LLM 实体/关系抽取（每 chunk 一次调用）
  └── 输出 ExtractedEntity[] + ExtractedRelation[]
  │
  ▼
_resolve_entity_aliases()
  ├── 字符级相似度打分
  ├── 联通分量分组
  ├── 选规范名、建映射
  └── 重定向关系、去重
  │
  ▼
KnowledgeGraphBuilder.build()
  ├── NetworkX 图构建（节点+边）
  └── 社群检测（Leiden/Louvain）
  │
  ▼
GraphStorage.store_graph()
  ├── 实体 → Milvus (kg_entities)
  ├── 关系 → Milvus (kg_relations)
  └── 社群 → Milvus (kg_communities)
```

### 调用方式

```python
from qa_core.knowledge_graph.pipeline import run_knowledge_graph_pipeline

result = await run_knowledge_graph_pipeline(
    chunks=docs,            # list[Document]
    kb_version="v1",
    collection_prefix="",   # 或 "novel_" 隔离数据集
    batch_size=10,
    max_gleanings=0,        # 0 = 仅一轮抽取
)
```

---

## 实体关系抽取

### LLM 输出格式

提词模版在 `qa_core/knowledge_graph/prompts.py`，要求 LLM 输出结构化文本：

**实体**：
```
("entity"<|>实体名<|>类型<|>描述)
```

**关系**（6 字段，含短标签）：
```
("relationship"<|>源实体<|>目标实体<|>短标签<|>详细描述<|>强度)
```

字段分隔符 `<|>`，记录分隔符 `##`，结束符 `<|COMPLETE|>`。

### 解析逻辑

解析器在 `extractor.py` 的 `_parse_response()`，**先尝试 6 字段新格式**，兼容旧 5 字段（自动从描述推断短标签）：

```python
if len(parts) >= 6:
    # 新格式：有 label 字段
    label = parts[3]
elif len(parts) >= 5:
    # 旧格式兼容：_auto_label_from_desc(desc) 推断
```

### 已知问题

| 问题 | 原因 | 对策 |
|------|------|------|
| 同人不同名 | LLM 在不同章节用了不同称呼 | 别名解析合并 |
| 实体类型不一致 | 同一事物被归类不同（"焚决"=概念/功法） | 合并时保留下级类型 |
| 描述过长 | 多章节合并导致描述叠加 | UTF-8 安全截断 4000 字节 |

---

## 短标签生成

关系边显示在图上时，需要短标签（2~6 字）而非完整描述。实现方式：

### 方案一：LLM 生成（首选）

提词要求 LLM 输出 6 字段格式，第 4 字段为短标签。

### 方案二：自动推断（兜底）

当 LLM 输出旧 5 字段格式时，`_auto_label_from_desc()` 从描述文本推断：

**规则优先级**：
1. `"X是Y的Z"` 模式 → 提取 Z（"天蚕土豆是《斗破苍穹》的作者" → "作者"）
2. 去掉主语后的首动词/介词 → "修炼"、"所在地"、"关系"
3. 已知关系词匹配 → "敌对"、"师徒"、"父子"
4. 从句子结构提取 → "参加成年仪式" → "成年仪式"

### 前端显示

`graph.html` 使用 vis.js，边标签默认显示短标签，悬停显示完整描述：

```javascript
const edgeLabel = rawLabel.length <= 8 ? rawLabel : rawLabel.slice(0, 6) + '…';
// label → 边标签显示
// title → 悬停 tooltip 显示完整描述
```

---

## 别名解析（实体对齐）

### 为什么需要

实体级精确检索下，不同名称导致检索不完整：
- 搜"萧薰儿" → 找不到"薰儿"的关系
- 搜"黑色古戒" → 找不到"黑色戒指"的关系

### 算法流程

4 步走：

```
1. 两两打分  →  2. 图联通分量  →  3. 选规范名  →  4. 合并+重定向
```

### 相似度打分（`_sim` 函数）

按顺序执行，一旦命中立即返回：

```
1. 字符完全重排        → "米特尔"=="特米尔"              → 0.85
2. 安全阀：3字+同姓    → "纳兰桀"≠"纳兰肃"              → 0.00
   （中字同末字异=不同人，中字异末字同=别名）
3. 安全阀：尾缀同首字异 → "三年之约"≠"七年之约"         → 0.00
4. 安全阀：颜色首字     → "黑色卷轴"≠"红色卷轴"         → 0.00
5. 安全阀：等级后缀     → "斗师"≠"大斗师"               → 0.00
6. 同长度+Jaccard≥0.5  → "成人仪式"≈"成年仪式"          → 0.70
7. 短名是长名后缀       → "薰儿"←"萧薰儿"               → 0.85
8. Jaccard≥0.6+非层级前缀 → "纳兰家"→"纳兰家族"         → 0.70
   （层级前缀如"斗之气"→"斗之气旋"被排除）
```

### 联通分量分组

```python
alias_graph = {
    "萧薰儿": {"萧熏儿", "薰儿"},
    "萧熏儿": {"萧薰儿"},
    "薰儿":   {"萧薰儿"},
}
# DFS 遍历 → [{萧薰儿, 萧熏儿, 薰儿}, ...]
```

### 规范名选择

```python
FAMILY_NAMES = {'萧','纳','兰','米','特','加','列'}
# 优先选带姓氏的，其次选最长的
```

### 合并操作

```python
# 实体去重：同名合并描述
if name in merged:
    merged[name].description += f"；{e.description}"
else:
    merged[name] = ExtractedEntity(name=name, ...)

# 关系重定向：源/目标指向规范名
src = merged_to_canonical.get(r.source, r.source)
tgt = merged_to_canonical.get(r.target, r.target)
if src == tgt: continue  # 自环跳过
pair = (src, tgt)
if pair in seen: continue  # 去重
```

### 直接使用别名解析

也可以不跑完整管线，单独调用：

```python
from qa_core.knowledge_graph.pipeline import _resolve_entity_aliases
merged_entities, merged_relations = _resolve_entity_aliases(
    entities, relationships, enabled=True
)
```

### 效果（斗破苍穹 71 章）

| 指标 | 合并前 | 合并后 |
|------|--------|--------|
| 实体 | 201 | 170（-15.4%） |
| 关系 | 270 | 255（-5.6%） |
| 别名组 | — | 15 组 |

---

## Milvus 存储设计

### 三个集合

| 集合 | 前缀+名 | 用途 | 关键字段 |
|------|---------|------|---------|
| 实体 | `{prefix}kg_entities` | 实体节点 | name, type, description, description_vector(1024d) |
| 关系 | `{prefix}kg_relations` | 关系边 | source, target, label, description, strength |
| 社群 | `{prefix}kg_communities` | 社群分组 | community_id, entities(json), summary |

### 注意事项

1. **VARCHAR 长度限制**：`description` 字段 max_length=4096，中文 UTF-8 编码下约 1300 字。必须用字节截断而非字符截断：

```python
def _truncate_utf8(text: str, max_bytes: int = 4000) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    return encoded[:max_bytes].decode("utf-8", errors="ignore")
```

2. **向量占位**：关系集合和社群集合无语义向量，但有 `dummy_vec(2d)` 满足 Milvus 必须至少一个向量字段的约束。

3. **动态字段**：`enable_dynamic_field=True` 允许额外字段，但动态字段的 VARCHAR 默认长度可能很大，注意插数据时不要误写长字段。

---

## 可视化服务

### 架构

```
kg_server.py（FastAPI）
  ├── GET /graph        → 静态页面 graph.html（vis.js）
  └── GET /api/graph/query?q=搜索词&top_k=15
        → JSON {nodes: [{id, label, title, group, color, degree, size}],
                 edges: [{from, to, label, title, value, width}]}
```

### 关键设计

**二跳展开**：搜到实体后，不仅返回直接邻居，还展开邻居的邻居：

```python
# 1. 搜实体
entities = client.query(..., filter=f'name like "%{q}%"')
# 2. 一阶邻居
for name in entity_names:
    rels = client.query(..., filter=f'source=="{name}" or target=="{name}"')
# 3. 补邻居实体
# 4. 二跳展开：对邻居再查关系
for neighbor in missing:
    hop2_rels = client.query(..., filter=f'source=="{neighbor}" or target=="{neighbor}"')
```

**实体/边去重**：同名实体只保留一个，同对关系合并描述：

```python
def _dedup_entities(raw): ...   # 按 name 去重
def _dedup_edges(raw): ...      # 按 (source,target) 有序对去重
```

**层级渲染**：按连接度分 4 层（叶子/次要/重要/核心），核心节点大字白框、叶子节点半透明小字。

### 启动方式

```bash
# 小说图谱（端口 18080）
KG_PREFIX=novel_ KG_PORT=18080 python kg_server.py

# MedicalKG 图谱（另一个端口）
KG_PORT=18081 python kg_server.py
```

---

## 与 Microsoft GraphRAG 对比

### 核心差异

| 维度 | 本实现 | Microsoft GraphRAG |
|------|--------|-------------------|
| 检索粒度 | **实体级** | **社群级** |
| 别名合并 | 需要（否则漏检） | 不需要（社群摘要包容） |
| 回答精度 | 精确可追溯 | 综合泛化 |
| 构建成本 | 低（1 轮 LLM 抽取 + 规则） | 高（多轮 LLM 摘要生成） |
| 存储 | Milvus（纯向量库） | 文本（社群摘要） |
| 查询方式 | 搜实体→展开邻居 | 匹配社群摘要→LLM 综合 |
| 可视化 | 力导向图（vis.js） | 通常是摘要视图 |

### 为什么微软不需要别名合并

微软 GraphRAG 不走实体匹配，流程是：

```
文档 → LLM 抽取实体/关系 → 构建图 → Leiden 社群检测
  → 对每个社群生成摘要（"这个社群包含XXX人物，主要关系是XXX"）
  → 用户提问 → 找语义相关的社群摘要 → 用摘要+原始文本回答
```

当用户问"萧薰儿的药"时，系统匹配的是**包含萧薰儿的那个社群的摘要**，摘要里已经写了"萧薰儿（又称熏儿）是萧炎的青梅竹马"。**不需要精确匹配实体名**，所以 "萧薰儿" 和 "熏儿" 是否合并不影响检索结果。

**代价**：社群摘要是泛化的，丢失细节。如果用户问"萧薰儿什么修为"，社群摘要可能只写了"萧薰儿是年轻一辈第一人"，不包含具体修为等级。

### 本实现的取舍

选择实体级检索的原因：

1. **可追溯**：用户可以看到"为什么回答了这个问题"——"因为搜到了这个实体和这些边"
2. **可交互**：力导向图可以探索，点一个实体看它和谁有关系
3. **精确性**：适合需要准确答案的场景（"萧炎用的是什么功法"）
4. **构建成本低**：不需要每社群的 LLM 摘要生成

不适用场景：
- 需要综合多个实体/篇章才能回答的问题 → GraphRAG 的社群摘要更好
- 模糊查询（"给个小说概要"）→ GraphRAG 的社群级回答更合适

### 两种模式共存的可能性

当前架构其实可以同时支持两种模式：

```
local_search.py 做社群语义匹配（GraphRAG 风格）
kg_server.py 做精确实体图探索（精确检索风格）
```

用户问题时可以先试精确实体检索，分数低再走社群摘要——两个模式互补。

### 别名解析的替代方案

如果不做别名合并，还可以：
1. **模糊查询**：`name like "%薰%"`（但会匹配到不需要的实体）
2. **同义词表**：手动维护 `{"萧薰儿": ["薰儿", "萧熏儿"]}`（维护成本高）
3. **向量召回**：用实体名 embedding 做语义匹配（需要额外向量检索）
4. **LLM 实时判断**：查询时让 LLM 展开别名（每次查询额外 LLM 调用）

别名合并是**构建时一次处理**，性价比最高的方案。
