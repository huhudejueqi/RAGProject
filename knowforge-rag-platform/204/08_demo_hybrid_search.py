# -*- coding: utf-8 -*-
# ============================================================================
# 第 08 章 演示脚本：Milvus 混合检索与 FAQ/Doc 分层检索
# ============================================================================
# 这是一个命令行交互脚本，演示第 08 章核心功能：
#   1. search_many() — Milvus 单 collection 内 dense 向量 + BM25 sparse 混合召回
#   2. FAQ/Doc 分层检索 — 由 RetrievalPlan.run_faq/run_doc 控制两路检索
#
# 术语边界（本章关键概念区分）：
#   - "Milvus Hybrid Search" 指单个 collection 内的 dense + BM25 sparse 混合召回
#   - "FAQ collection + Doc collection" 是两路/分层检索，不是 hybrid search
#   - 两路由第 06 章生成的 RetrievalPlan.run_faq/run_doc 分别控制
#
# 本脚本展示从用户输入到检索结果的完整调用链（不生成最终回答）：
#
#   1. decide_route()         — 入口路由（直答/source 边界/检索）
#   2. classify_intent()      — 意图分类（FAQ_QUERY / KNOWLEDGE_QUERY 等）
#   3. rewrite_query_if_needed() — 追问改写（如 "那审批呢" → 完整问题）
#   4. build_retrieval_plan() — 生成检索计划（top_k、source、rerank 等参数）
#   5. generate_query_variants() — 生成查询变体（同义词、等价表达）
#   6. search_many()          — 按 RetrievalPlan 分别或同时检索 FAQ/文档库
#
# 这是第 05~08 章所有模块首次协同工作的演示。
#
# 用法示例：
#   # 基础知识查询 — FAQ + Doc 两路检索
#   python scripts\demo_hybrid_search.py "入职流程是什么"
#
#   # 指定 source 过滤 — 只查 finance 来源
#   python scripts\demo_hybrid_search.py "报销" --source finance
#
#   # 追问 + 历史 — 测试追问改写后的检索效果
#   python scripts\demo_hybrid_search.py "那审批呢" --history "报销流程是什么"
#
#   # 指定知识库版本和数据域 — 不指定版本时使用当前 active 版本
#   python scripts\demo_hybrid_search.py "账号回收" --kb-version kb_v2 --visibility internal
#
#   # 多角色用户 — 演示 allowed_roles 过滤
#   python scripts\demo_hybrid_search.py "VPN" --role admin --role public
#
# 输出格式：缩进美化的 JSON，包含 query、rewritten_query、kb_version、
# query_variants、retrieval_plan、faq_sources、doc_sources
# ============================================================================

# from __future__ import annotations 使所有类型注解延时求值
# 好处：允许在类型注解中使用尚未定义的类名，且运行时不会真正求值
"""演示 FAQ 与文档分集合的 Milvus 混合检索。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

# argparse: Python 标准命令行参数解析库
#   提供位置参数(query)和多个可选参数的解析
import argparse

# json: 标准 JSON 序列化库
#   json.dumps(..., ensure_ascii=False) 保证中文正常显示
#   json.dumps(..., indent=2) 提供可读的缩进格式
import json

# sys: 系统相关功能
#   sys.path.insert(0, ...) 将本章目录加入 Python 模块搜索路径
import sys

# pathlib.Path: 面向对象的文件路径操作
#   Path(__file__).resolve().parents[1] 获取本章根目录的绝对路径
from pathlib import Path

# HumanMessage: LangChain 标准人类消息类型
#   用于存储历史对话中的用户消息
from langchain_core.messages import HumanMessage

# ── 路径设置 ──
# 当从任意目录运行此脚本时，需要确保 Python 能找到 qa_core 包。
# __file__           → scripts/demo_hybrid_search.py 的路径（可能是相对路径）
# .resolve()         → 转为绝对路径
# .parents[1]        → 向上两层，即 ch08_milvus_hybrid_search/ 目录
# sys.path.insert(0) → 将此目录插入模块搜索路径最前面
CHAPTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPTER_DIR))

from qa_core.storage.bootstrap import bootstrap_mysql_schema  # noqa: E402

# ── 导入核心 API ──
# 以下 noqa: E402 注释告诉 flake8 忽略"import 不在文件顶部"的警告
# — 因为 sys.path 必须在 import 之前修改

# DataScope: 数据域隔离五元组（tenant_id / dataset_id / visibility / user_roles）
from qa_core.governance.data_scope import DataScope  # noqa: E402

# resolve_active_kb_version: 获取当前场景的 active 知识库版本
from qa_core.governance.kb_versions import resolve_active_kb_version  # noqa: E402

# classify_intent: 意图分类统一入口（复用第 05 章）
from qa_core.intent.classifier import classify_intent  # noqa: E402

# generate_query_variants: 查询变体生成（复用第 07 章）
from qa_core.pipeline.query_variants import generate_query_variants  # noqa: E402

# rewrite_query_if_needed: 追问改写（复用第 07 章）
from qa_core.pipeline.rewrite import rewrite_query_if_needed  # noqa: E402

# decide_route: 路由决策统一入口（复用第 05 章）
from qa_core.pipeline.query_input import normalize_user_query
from qa_core.pipeline.steps import decide_route  # noqa: E402

# resolve_scenario: 获取当前业务场景配置
from qa_core.scenarios.registry import resolve_scenario  # noqa: E402

# get_doc_store / get_faq_store: 获取 Doc/FAQ 检索 Store 实例（单例模式）
#   每个 Store 内部封装了 Milvus dense + BM25 sparse 混合检索
from qa_core.retrieval.factory import get_doc_store, get_faq_store  # noqa: E402

# RetrievalResult: 检索结果对象，包含 source_payloads() 等方法
from qa_core.retrieval.results import RetrievalResult  # noqa: E402

# build_retrieval_plan: 检索计划生成（复用第 06 章）
from qa_core.retrieval.strategy import build_retrieval_plan  # noqa: E402


def main() -> None:
    """解析命令行参数 → 路由 → 意图 → 改写 → 计划 → 变体 → 数据域 → 混合检索 → 输出 JSON。

    执行流程（完整串联第 05~08 章）：
      1. 解析命令行参数（query 必传，其他均为可选）
      2. 调用 decide_route(query) — 第 05 章路由决策
      3. 如果命中直答规则（问候/转人工/越界/source边界）→ 直接返回直答结果
      4. classify_intent(query, history, scenario) — 第 05 章意图分类
      5. rewrite_query_if_needed(query, history, requires_rewrite) — 第 07 章改写
      6. build_retrieval_plan(rewritten_query, intent) — 第 06 章检索计划
      7. generate_query_variants(rewritten_query, ...) — 第 07 章变体生成
      8. 构造 DataScope 数据域隔离参数
      9. 按 plan.run_faq / plan.run_doc 分别执行 FAQ/Doc 混合检索
      10. 汇总所有中间产物和检索结果输出 JSON

    调用顺序：章节流程 -> main()。
    """

    # 入口显式初始化 MySQL schema，业务 Store 不做按需建表。
    bootstrap_mysql_schema()

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 0：构建命令行参数解析器
    # ════════════════════════════════════════════════════════════════════════
    parser = argparse.ArgumentParser(
        description="第 08 章：Milvus Hybrid Search 与 FAQ/Doc 分层检索演示"
    )

    # 位置参数：用户问题（必传）
    # argparse 自动将第一个非 - 开头的参数绑定到此
    parser.add_argument("query", help="用户问题")

    # 可选参数：历史对话列表
    # nargs="*": 接受 0 个或多个值 → list[str]
    # default=None: 未传时为 None
    parser.add_argument(
        "--history", nargs="*", default=None,
        help='可选历史问题列表，用于演示追问检索（如：--history "报销流程是什么"）'
    )

    # 可选参数：source 过滤
    # 限制检索的数据来源（如 finance/hr/it），覆盖 intent 的 suggested_source
    parser.add_argument("--source", default=None, help="source_filter，如 finance/hr/it")

    # 可选参数：知识库版本
    # 不传时使用当前 active 版本；传了则强制使用指定版本
    parser.add_argument("--kb-version", default=None, help="知识库版本；默认使用当前 active 版本")

    # ── 数据域隔离参数 ──
    # 这些参数共同构成 DataScope 五元组，用于 Milvus 过滤表达式
    parser.add_argument("--tenant-id", default="default", help="租户 ID（数据隔离的第一维度）")
    parser.add_argument("--dataset-id", default="default", help="数据集 ID")
    parser.add_argument(
        "--visibility", default="public",
        help="文档可见性：public（所有人）/ internal（同租户）/ private（指定角色）"
    )

    # action="append": 每次指定的 --role 值追加到 roles 列表
    # dest="roles": 存储到 args.roles 而不是 args.role
    parser.add_argument(
        "--role", action="append", dest="roles", default=None,
        help="用户角色（可多次指定，如 --role admin --role hr）"
    )

    args = parser.parse_args()

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 1：入口路由 — 判断是否需要检索（第 05 章）
    # ════════════════════════════════════════════════════════════════════════
    # 将历史消息列表转为 LangChain HumanMessage 对象
    # args.history or []: 当 --history 未传时为 None，fallback 到空列表
    history = [HumanMessage(content=item) for item in args.history or []]

    # resolve_scenario() 返回当前活跃的业务场景配置
    # 场景配置包含：valid_sources、faq_collection、doc_collection 等
    scenario = resolve_scenario()

    # decide_route() 执行路由决策：
    # - 如果问题匹配了直答规则（问候/转人工/越界/source边界），route.answer 非空
    # - 本脚本只在命中直答规则时直接返回，否则继续检索链路
    effective_query = normalize_user_query(args.query)
    route = decide_route(effective_query, scenario=scenario, source_filter=args.source)

    # ── 直答快路径：如果路由决策给出了直接答案，立即返回 ──
    if route.answer:
        payload = {
            "query": args.query,
            "route": route.route,
            "answer": route.answer,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 2：意图分类 — 判断问题类型（第 05 章）
    # ════════════════════════════════════════════════════════════════════════
    # classify_intent() 返回 IntentResult，包含：
    #   - intent: FAQ_QUERY / KNOWLEDGE_QUERY / FOLLOW_UP 等
    #   - suggested_source: 推断的数据来源（如 "finance"）
    #   - requires_rewrite: 是否需要追问改写
    intent = classify_intent(effective_query, history=history, scenario=scenario)

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 3：追问改写 — 将依赖上下文的问题独立化（第 07 章）
    # ════════════════════════════════════════════════════════════════════════
    # rewrite_query_if_needed() 行为：
    #   - intent.requires_rewrite=True  → 根据历史补全追问（如 "那审批呢" → "报销的审批流程是什么"）
    #   - intent.requires_rewrite=False → 原样返回 args.query
    rewritten_query = rewrite_query_if_needed(
        effective_query, history, intent.requires_rewrite
    )

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 4：检索计划 — 根据意图生成检索参数（第 06 章）
    # ════════════════════════════════════════════════════════════════════════
    # build_retrieval_plan() 返回 RetrievalPlan，包含 15 个检索参数：
    #   run_faq / run_doc: 是否执行 FAQ/Doc 检索
    #   faq_top_k / doc_top_k: 召回数量
    #   use_query_variants: 是否生成查询变体
    #   rerank: 是否启用重排序
    plan = build_retrieval_plan(rewritten_query, intent)

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 5：查询变体 — 生成等价表达以提升召回（第 07 章）
    # ════════════════════════════════════════════════════════════════════════
    # generate_query_variants() 行为：
    #   - enabled=True → 生成同义词/等价表达变体
    #   - enabled=False → 返回只含原问题的单元素列表
    #   - allow_short_structured=FOLLOW_UP: 追问场景允许对短结构化问题生成变体
    query_variants = generate_query_variants(
        rewritten_query,
        enabled=plan.use_query_variants,
        allow_short_structured=intent.intent == "FOLLOW_UP",
    )

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 6：构造数据域隔离参数
    # ════════════════════════════════════════════════════════════════════════
    # DataScope 五元组决定检索时能看到哪些文档：
    #   - tenant_id: 租户隔离（不同租户数据完全不可见）
    #   - dataset_id: 数据集隔离
    #   - visibility: public / internal / private
    #   - user_roles: 角色列表（用于 allowed_roles 匹配）
    data_scope = DataScope(
        tenant_id=args.tenant_id,
        dataset_id=args.dataset_id,
        visibility=args.visibility,
        user_roles=args.roles or ["public"],
    )

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 7：按检索计划执行 FAQ/Doc 两路检索
    # ════════════════════════════════════════════════════════════════════════
    # 检索范围：企业知识库场景预置了 hr、finance、it 三个来源
    valid_sources = scenario.valid_sources

    # 获取当前 active 知识库版本（如果没有显式指定 --kb-version）
    # resolve_active_kb_version(None, scenario_id) 返回该场景当前的 active 版本
    active_kb_version = args.kb_version or resolve_active_kb_version(None, scenario.scenario_id)

    # ── 7a. FAQ 检索路 ──
    # FAQ/Doc 是两路检索；每一路内部的 search_many() 才是 Milvus dense + BM25 sparse hybrid search
    # 企业问答默认两路都查，但真实业务不能写死，必须服从第 06 章生成的 run_faq/run_doc
    if plan.run_faq:
        # get_faq_store() 返回 FAQ collection 的检索 Store 单例
        # search_many() 参数说明：
        #   query_variants: 多个查询变体，内部对每个变体执行 hybrid search 然后合并去重
        #   k=plan.faq_top_k: 从检索计划中取 FAQ 路召回数量
        #   source_filter: 优先使用命令行指定的 source，否则用 intent 推断的 source
        #   kb_version: 只检索指定版本的数据
        #   valid_sources: 场景白名单（过滤掉不在白名单内的 source）
        #   data_scope: 数据域隔离参数（tenant/visibility/role）
        #   source_type="faq": 标记来源类型（用于结果中区分 FAQ/Doc）
        #   rerank=plan.rerank: 是否启用重排序
        faq_result = get_faq_store().search_many(
            query_variants,
            k=plan.faq_top_k,
            source_filter=args.source or intent.suggested_source,
            kb_version=active_kb_version,
            data_scope=data_scope,
            source_type="faq",
            rerank=plan.rerank,
        )
    else:
        # ── FAQ 路被检索计划跳过 ──
        # 返回空的 RetrievalResult，但保留查询信息便于输出
        faq_result = RetrievalResult(
            query=" | ".join(query_variants),
            source_type="faq",
        )

    # ── 7b. Doc 检索路 ──
    if plan.run_doc:
        # get_doc_store() 返回 Doc collection 的检索 Store 单例
        # 参数含义与 FAQ 路完全一致，但 source_type="doc"
        doc_result = get_doc_store().search_many(
            query_variants,
            k=plan.doc_top_k,
            source_filter=args.source or intent.suggested_source,
            kb_version=active_kb_version,
            data_scope=data_scope,
            source_type="doc",
            rerank=plan.rerank,
        )
    else:
        # ── Doc 路被检索计划跳过 ──
        doc_result = RetrievalResult(
            query=" | ".join(query_variants),
            source_type="doc",
        )

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 8：汇总输出 — 所有中间产物和检索结果
    # ════════════════════════════════════════════════════════════════════════
    # source_payloads() 返回检索结果的精简表示（去掉向量和内部字段）
    payload = {
        "query": args.query,                    # 用户原始输入
        "rewritten_query": rewritten_query,     # 第 07 章改写后的问题
        "kb_version": active_kb_version,        # 实际使用的知识库版本
        "query_variants": query_variants,       # 第 07 章生成的查询变体列表
        "retrieval_plan": plan.as_dict(),       # 第 06 章检索计划的完整 15 字段
        "faq_sources": faq_result.source_payloads(),  # FAQ 检索召回结果
        "doc_sources": doc_result.source_payloads(),  # Doc 检索召回结果
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 当脚本直接运行时（python demo_hybrid_search.py），__name__ == "__main__"
    # — 此时 main() 不被调用
    main()
