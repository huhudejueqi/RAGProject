# KnowForge RAG Platform V1.0.10 Final Release

这是一个基于 `LangChain + Milvus Hybrid Search + FastAPI` 的多场景企业级 RAG 系统项目。项目目标不是做一个简单聊天页面，而是把企业级 RAG 的主链路、知识库治理、RAG 回归验收、版本管理、数据隔离、三级缓存和流式问答做成可以演示、可以验收、可以写进简历的完整工程。

当前业务场景已经冻结为 8 个，一期不再继续新增场景包；后续重点放在资料质量、评测回归、版本治理和产品化体验。**一期源码不提前放 Agent 预留实现**，本发布包已经剔除二期 Agent、GraphRAG、外部协议和任务队列代码。

## 1. 项目定位

一句话介绍：

> 基于 LangChain 和 Milvus Hybrid Search 构建的 KnowForge RAG Platform，支持 FAQ 直出、文档问答、知识库多版本、引用式增量、数据隔离、三级缓存、流式输出、入库质量检查和 RAG 回归验收。

## 2. 业务场景

| 场景 ID | 业务背景 | source 数 | FAQ | 文档 | 简历包装 |
| --- | --- | ---: | ---: | ---: | --- |
| `enterprise_knowledge` | HR、IT、财务制度 | 3 | 8 | 11 | 企业内部知识库智能问答平台 |
| `saas_support` | 账号、计费、开放集成 | 3 | 6 | 11 | SaaS 客服知识库智能助手 |
| `equipment_ops` | 巡检、告警、安全规范 | 3 | 6 | 11 | 制造业设备运维知识助手 |
| `compliance_qa` | 合同、审计、隐私保护 | 3 | 6 | 11 | 企业合规制度智能问答系统 |
| `cross_border_risk` | 海关、制裁、信用证、物流、单证 | 5 | 11 | 15 | 跨境贸易风控 RAG 知识问答平台 |
| `tender_contract_risk` | 招投标、合同、交付、验收、履约风险 | 5 | 11 | 15 | 招投标合规与合同履约 RAG 风控平台 |
| `insurance_claims` | 保单、理赔材料、责任、除外、赔付 | 5 | 10 | 15 | 保险理赔材料审核与 RAG 知识问答助手 |
| `engineering_project_qa` | 图纸、规范、进度、质量、安全资料 | 5 | 11 | 15 | 工程项目资料与施工规范 RAG 问答助手 |

## 3. V1 核心闭环

| 能力 | 当前实现 |
| --- | --- |
| 多场景切换 | `scenarios/<scenario_id>/scenario.toml + faq.csv + data/` 配置化切换 |
| 混合检索 | Milvus dense vector + Milvus 内置 BM25 sparse |
| FAQ 直出 | 高置信 FAQ 直接返回标准答案，低置信进入文档 RAG |
| 文档 RAG | LangChain loader/splitter + parent-child chunk + rerank |
| 意图识别 | 规则优先 + 可选小模型决策网关，分数进入检索计划闭环 |
| 知识库版本 | STAGED / ACTIVE / ARCHIVED、active 指针、回滚流水、引用式增量有效期窗口 |
| 数据隔离 | tenant、dataset、visibility、allowed_roles 写入 metadata 并参与检索过滤 |
| 三级缓存 | L1 进程内 epoch 缓存、L2 Redis 检索/embedding 缓存、L3 MySQL cache namespace 治理 |
| 质量门禁 | 入库质量报告、FAQ/正文冲突检测、低质量 chunk 检测、表格/OCR 风险统计 |
| 评测回归 | Recall@K、MRR、关键词覆盖、模板命中率、场景隔离率、表格行召回专项回归 |
| Bad Case 闭环 | 本地评测报告 + `extract_bad_cases_from_report.py` + `eval_sets/`，人工确认后进入回归评测 |
| 观测诊断 | 阶段耗时、首 token、检索诊断、来源引用、answer_confidence、LangSmith 可选 Trace |

## 4. 目录说明

| 目录 | 说明 |
| --- | --- |
| `app.py` | FastAPI 应用入口，只注册 V1 页面、聊天、管理和知识库版本路由 |
| `qa_core/` | V1 RAG 主链路、入库、治理、缓存、评测和观测代码 |
| `static/` | V1 问答页和状态页 |
| `docs/` | V1 讲义 Markdown 源文件 |
| `site/` | 已构建的网页版讲义，入口为 `site/index.html` |
| `codealong/` | V1 跟敲代码 |
| `scenarios/` | 8 大冻结业务场景样例数据 |
| `eval_sets/` | V1 回归评测和性能基线样例 |
| `scripts/` | V1 入库、评测、缓存验收、Docker 验收和发布验收脚本 |
| `tests/` | V1 单测与回归测试 |

源码注释覆盖门禁：

```powershell
python scripts/check_source_comment_coverage.py --fail-on-issues
```

该门禁要求每个 V1 Python 文件具备模块职责说明、公开类和函数具备 docstring，且每个自研前端 JavaScript 文件具备职责与交互边界说明。

## 5. 常用命令

```powershell
python -m mkdocs build --strict
python scripts/verify_v1_release.py
python scripts/run_v1_quality_cycle.py --docker --include-performance
python scripts/calibrate_thresholds.py --fail-on-insufficient
```

Docker 本地部署：

```powershell
# 先按 models/README.md 将三个模型目录放到 ./models
# 本包已提供 .env.compose，再配置可用 API Key
notepad .env.compose
docker compose --env-file .env.compose up -d --build
```

Linux 使用相同目录结构和命令。Compose 默认挂载 `./models:/app/models`，无需在配置中写 Windows 或 Linux 的绝对路径。

新环境初始化 8 大场景：

```powershell
docker compose --env-file .env.compose run --rm api python scripts/rebuild_scenarios.py --reset-collections --description "v1 init all scenarios"
```

缓存验收：

```powershell
docker compose --env-file .env.compose exec api python scripts/cache_acceptance_smoke.py --base-url http://127.0.0.1:8000
```

网页版讲义：

```text
site/index.html
```
