# medical_assistant 场景

第 9 个业务场景，面向“医院服务 + 疾病百科 + 药品资料”的医疗 RAG 问答。

## 当前数据

- `faq.csv`：93 条标准问答，覆盖医院服务、常见疾病和药品。
- `data/hospital_data/`：4 份医院资料。
- `data/disease_data/`：91 份高频常见疾病 Markdown。
- `data/drug_data/药品数据.md`：4525 条药品适应症/不良反应数据，来自 Milvus `demo_medical_docs` 集合；已过滤疾病、症状和手术条目，并按 `## 药品名` 保留独立章节。

当前已激活版本：

```text
kb_medical_assistant_20260804_112939_d2b4273a
```

## 构建与启动

```bash
# 首次构建（本场景已经执行过）
KNOWLEDGE_GRAPH_ENABLED=false \
python scripts/rebuild_kb_version.py \
  --scenario medical_assistant \
  --new-version \
  --reset-collections \
  --activate \
  --max-low-quality-issues 3000 \
  --max-duplicate-chunks 600 \
  --max-faq-document-conflicts 100

# 药品数据清洗后的增量重建
KNOWLEDGE_GRAPH_ENABLED=false \
python scripts/rebuild_kb_version.py \
  --scenario medical_assistant \
  --new-version \
  --incremental-from active \
  --activate \
  --description "药品数据仅保留药品章节，来源标签改为具体药品名" \
  --max-low-quality-issues 3000 \
  --max-duplicate-chunks 600 \
  --max-faq-document-conflicts 100

# 以本场景作为前端默认场景启动
ACTIVE_SCENARIO_ID=medical_assistant KNOWLEDGE_GRAPH_ENABLED=false python app.py
```

打开 `http://localhost:18000/` 后，顶部场景下拉框选择“医疗知识助手”。

## 验证问题

- 百日咳有哪些症状？
- 肺炎怎么治疗？
- 原发性高血压需要做哪些检查？
- 什么药可以缓解咳嗽？
- 初诊患者如何挂号？
- 腹部不适

## 来源展示

Markdown 文档的来源标签优先显示章节标题，因此 `药品数据.md` 命中多条时，参考来源会显示具体药品名，例如“活血止痛片”“大明胶囊”“热毒清片”，而不是三条都叫“药品数据.md”。

每条回答下方会提供一个“知识图谱”链接，以当前问题打开 `/graph?q=...`。医疗场景当前仍以 `KNOWLEDGE_GRAPH_ENABLED=false` 运行，所以链接用于查看图谱可视化，不会把旧图谱数据注入 RAG 答案。

## 质量门禁说明

医疗百科本身包含大量短字段（治疗方式、科室、检查项等），部分医院 FAQ 也没有一一对应到正文片段。因此本次激活显式放宽了低质量 chunk、重复 chunk 和 FAQ 文档冲突阈值；放宽值都在 README 和构建命令中可见，正式使用前应继续清洗短字段和补齐 FAQ 依据。

知识图谱默认未构建也未注入本次 RAG 问答，避免旧图谱数据污染医疗结果。后续若要为医疗数据构建独立图谱，可先清空/隔离 `kg_*` 集合，再使用 `KNOWLEDGE_GRAPH_ENABLED=true` 重建。

## 全量疾病数据

当前场景放入的是 91 个高频疾病，全量 8808 份疾病文档在：

```text
data_packs/medical_disease_data/docs/
```

如需全量替换，可将该目录全部复制到 `scenarios/medical_assistant/data/disease_data/`，再按新版本构建流程重新入库。全量数据量较大，建议分批或按业务需要筛选后再跑。
