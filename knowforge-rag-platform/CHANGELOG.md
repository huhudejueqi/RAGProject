# Changelog

本文件记录仓库阶段版本，不属于课程讲义正文。

## v1.0.11

发布日期：2026-07-21

### 工程清理补丁

- 删除已被标准 Compose 完全替代的 `docker-compose.milvus.yml`，部署入口统一为 `docker-compose.yml`。
- 删除未进入课程导航且已被正式章节动画替代的内部流程页 `docs/animation/codealong-code-flow.html`。
- 本地运行诊断只检查标准 Compose，避免旧端口和旧容器名称干扰排障。
- 发布守卫禁止上述历史文件重新进入 V1 发布物。
- 不改变业务功能、API、数据库结构、检索参数和知识库数据。

## v1.0.10

发布日期：2026-07-21

### 注释与可维护性补丁

- 为 V1 主项目、运维脚本、测试和章节跟敲代码补齐模块职责与公开接口 docstring。
- 为自研前端 JavaScript 文件补齐职责、协议和交互边界说明。
- 新增 `scripts/check_source_comment_coverage.py`，把注释覆盖纳入封版门禁。
- 不改变业务功能、API、数据库结构、检索参数和知识库数据。

## v1.0.9

发布日期：2026-07-21

### 定版内容

- 冻结 V1 多场景企业级 RAG 主链路、知识库版本治理和引用式增量实现。
- 完成 BERT 意图模型网关、策略评测、模型治理报告和风险保护闭环。
- 完成 L1/L2/L3 三级缓存、版本激活失效和缓存验收。
- 完成文档图片/OCR 风险识别与人工复核边界。
- 完成评测、性能门禁、Bad Case 草稿、无人值守质量周期和阈值候选校准。
- 修复第三方 LLM 原始错误直接暴露到聊天页面的问题。
- 更新讲义、跟敲代码、网页版站点和 Docker 发布结构。

### 验收边界

代码、Docker、检索、意图模型、讲义和离线门禁独立验收。DashScope 当前账户欠费，因此最终在线生成评测必须在账户恢复后重新执行；发布包不会把该外部失败伪装为通过。

## v1.0.0

发布日期：2026-06-25

### 阶段定位

V1.0 稳定交付版。该版本作为后续 V1.0 维护和 V2.0 开发的正式基线。

### 核心内容

- 完成多场景企业级 RAG 主链路：FAQ 直出、文档 RAG、混合检索、Prompt Profile、流式问答和引用返回。
- 完成 8 个业务场景的数据包、场景配置和知识库版本激活流程。
- 完成 MySQL 知识库版本元数据、active 指针、版本质量门禁和回归验证脚本。
- 完成多格式文档加载，统一依赖到 `requirements.txt`，并接入 Docling 作为增强解析后端。
- 完成 `codealong/` 章节实操代码与主项目核心链路对齐。
- 完成讲义、独立链路图、静态站点、Docker Compose 部署和项目守卫脚本对齐。

### 验证结果

- `python scripts/check_project_guardrails.py` 通过
- `python -m unittest discover -s tests` 通过
- `python -m unittest discover -s codealong\chapters\ch16_ingestion_pipeline\tests` 通过
- `python scripts\check_codealong_alignment.py` 通过
- `python codealong\check_alignment.py` 通过
- `python -m mkdocs build` 通过
- `docker compose --env-file .env.compose build api` 通过
- `docker run --rm knowforge-rag-platform-api:latest python scripts/tools/docling_parser_smoke.py` 通过

### 后续维护

- V1.0 维护分支：`release/1.0`
- V2.0 开发分支：`develop/2.0`
- V1.0 修复验证后同步到 `main`，并按需合并或 cherry-pick 到 `develop/2.0`。

## v1.1.0-codealong-complete

发布日期：2026-06-12

### 阶段定位

完成 `codealong/` 跟敲型项目闭环。当前版本适合用于从第 05 章开始，按章节逐步实现企业级 RAG 项目的核心链路。

### 新增内容

- 新增 `codealong/chapters/ch05_intent_classification`
- 新增 `codealong/chapters/ch06_retrieval_strategy`
- 新增 `codealong/chapters/ch07_query_rewrite_variants`
- 新增 `codealong/chapters/ch08_milvus_hybrid_search`
- 新增 `codealong/chapters/ch09_qaservice_orchestration`
- 新增 `codealong/chapters/ch10_rag_pipeline`
- 新增 `codealong/chapters/ch11_prompt_engineering`
- 新增 `codealong/chapters/ch12_fastapi_service`
- 新增 `codealong/chapters/ch13_preflight_checks`
- 新增 `codealong/chapters/ch14_kb_versioning`
- 新增 `codealong/chapters/ch15_data_isolation`
- 新增 `codealong/chapters/ch16_ingestion_pipeline`
- 新增 `codealong/chapters/ch17_quality_evaluation`
- 新增 `codealong/chapters/ch18_test_system`
- 新增 `codealong/chapters/ch19_observability_tracing`
- 新增 `codealong/run_all_tests.py`
- 新增 GitHub Actions：`.github/workflows/codealong-ci.yml`
- 新增 `pytest.ini`，保证本机直接运行 `pytest tests -q` 时能稳定导入 `qa_core`

### 验证结果

- `python codealong\run_all_tests.py` 通过
- `pytest tests -q` 通过：66 passed，13 subtests passed
- `python scripts\check_project_guardrails.py` 通过
- `python scripts\check_no_polyfill_io.py` 通过
- `docker compose --env-file .env.compose config --quiet` 通过
- `python -m compileall -q codealong qa_core scripts app.py` 通过

### 维护说明

`codealong/` 是章节实操目录，由 Git 管理，但不会进入主项目 Docker 镜像。它保留课程需要亲手实现的主链路，不直接复制主项目完整生产级结构。

## v1.0.0-phase1-baseline

发布日期：2026-06-12

### 阶段定位

一期完整项目基线。包含多场景企业级 RAG 项目的主项目代码、讲义、动画、测试和 Docker 部署结构。

### 核心能力

- 多场景 RAG 问答
- FAQ 直出与文档 RAG 链路
- Milvus dense + sparse 混合检索
- source 推断与数据隔离
- 知识库版本管理
- 入库与质量检查
- 回归测试与评测脚本
- FastAPI 服务与前端页面
- Docker Compose 部署

### 维护说明

该标签保留为一期项目基线。后续一期修复走 `phase1-maintenance`，二期能力走 `phase2-graphrag`。
