# -*- coding: utf-8 -*-
# ============================================================================
# 第 09 章 演示脚本：QAService 核心编排
# ============================================================================
# 这是一个命令行交互脚本，演示第 09 章核心功能：
#   1. QAService.stream_query()  — 流式问答（逐 token 输出 SSE 事件）
#   2. QAService.debug_retrieval() — 检索诊断（只返回检索结果，不调用 LLM）
#
# 本章是跟敲项目的关键转折点 —— 从"零件"到"组装"。
# QAService 将前面四章构建的独立模块串联为一个完整的问答服务：
#
#   QAService 委托主链路后的核心调用：
#     ├─ decide_route()          → source 校验 + 直答 / FAQ 快路径 / 检索（第 05 章）
#     ├─ classify_intent()       → 检索意图分类（第 05 章）
#     ├─ build_retrieval_plan()  → 生成检索计划（第 06 章）
#     ├─ rewrite + variants      → 查询改写与变体生成（第 07 章）
#     ├─ search_faq + search_doc → 混合检索（第 08 章）
#     └─ build_context           → 构建 LLM 上下文（第 10 章起）
#
#   QAService 对外暴露两种模式：
#     stream_query()      → 流式问答：返回 SSE 风格事件列表
#                           [start, status, token, ..., end]
#     debug_retrieval()   → 检索诊断：只返回检索结果和中间产物
#                           不调用 LLM 生成回答，用于调试检索质量
#
# 用法示例：
#   # 流式问答 — 完整 RAG 链路
#   python scripts\demo_qa_service.py "入职流程是什么"
#
#   # 指定 source 过滤 — 限制检索数据来源
#   python scripts\demo_qa_service.py "报销流程" --source finance
#
#   # 检索诊断模式 — 不调用 LLM，只看检索结果
#   python scripts\demo_qa_service.py "入职需要哪些材料" --debug
#
#   # 指定会话 ID — 用于多轮对话历史管理
#   python scripts\demo_qa_service.py "那审批呢" --session-id my-session --source finance
#
#   # 指定知识库版本
#   python scripts\demo_qa_service.py "VPN" --kb-version kb_v2
#
# 输出格式：
#   - stream_query 模式：缩进美化的 JSON 事件数组
#     [{"type": "start", ...}, {"type": "status", ...}, {"type": "token", ...}, {"type": "end", ...}]
#   - debug 模式：缩进美化的 JSON 检索诊断对象
# ============================================================================

# from __future__ import annotations 使所有类型注解延时求值
# 好处：允许在类型注解中使用尚未定义的类名，且运行时不会真正求值
"""演示 QAService 统一入口的同步诊断和流式问答。

调用顺序：课程示例、测试或命令行入口 -> 本模块公开接口。
"""

from __future__ import annotations

# argparse: Python 标准命令行参数解析库
#   提供位置参数(query)和可选参数(--source, --session-id, --kb-version, --debug)的解析
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

# ── 路径设置 ──
# 当从任意目录运行此脚本时，需要确保 Python 能找到 qa_core 包。
# __file__           → scripts/demo_qa_service.py 的路径（可能是相对路径）
# .resolve()         → 转为绝对路径
# .parents[1]        → 向上两层，即 ch09_qaservice_orchestration/ 目录
# sys.path.insert(0) → 将此目录插入模块搜索路径最前面
CHAPTER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CHAPTER_DIR))

from qa_core.storage.bootstrap import bootstrap_mysql_schema  # noqa: E402

# ── 导入核心 API ──
# get_qa_service: 获取 QAService 单例（工厂模式）
#   返回全局唯一的 QAService 实例，内部已装配好所有依赖
#   noqa: E402 — sys.path 必须在 import 之前修改
from qa_core.application.factory import get_qa_service  # noqa: E402


def main() -> None:
    """解析命令行参数 → 获取 QAService → stream_query / debug_retrieval → 输出 JSON。

    执行流程：
      1. 解析命令行参数（query 必传，--source/--session-id/--kb-version/--debug 可选）
      2. 调用 get_qa_service() 获取 QAService 单例
      3. 根据 --debug 标志选择模式：
         a. debug_retrieval(query, source_filter, session_id, kb_version)
            → 返回检索诊断 dict（含意图、改写、计划、变体、检索结果）
         b. stream_query(query, source_filter, session_id, kb_version)
            → 返回 SSE 事件生成器，list() 收集全部事件
      4. 序列化为 JSON 并打印

    两种模式的对比：
      - stream_query: 走完整 RAG 链路，包含 LLM 生成回答和逐 token 事件流
      - debug_retrieval: 只走检索前链路（意图→改写→计划→变体→检索），
        不调用 LLM，适合调试检索质量和参数调优
    """

    # 入口显式初始化 MySQL schema，业务 Store 不做按需建表。
    bootstrap_mysql_schema()

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 1：构建命令行参数解析器
    # ════════════════════════════════════════════════════════════════════════
    parser = argparse.ArgumentParser(
        description="第 09 章：QAService 核心编排演示"
    )

    # 位置参数：用户问题（必传）
    # argparse 自动将第一个非 - 开头的参数绑定到此
    parser.add_argument("query", help="用户问题")

    # 可选参数：source 过滤
    # 限制检索的数据来源（如 finance/hr/it），传入 QAService 后内部会校验合法性
    parser.add_argument(
        "--source", default=None,
        help="source_filter，如 finance/hr/it"
    )

    # 可选参数：会话 ID
    # 用于管理多轮对话历史；同一 session-id 的请求可以共享对话上下文
    parser.add_argument(
        "--session-id", default="demo-session",
        help="会话 ID（用于对话历史管理，多次请求同 session-id 可继承上文）"
    )

    # 可选参数：知识库版本
    # 不传时 QAService 内部使用 resolve_active_kb_version() 获取当前 active 版本
    parser.add_argument(
        "--kb-version", default=None,
        help="知识库版本；默认使用当前 active 版本"
    )

    # 可选参数：检索诊断模式
    # action="store_true": 当 --debug 出现时 args.debug=True，否则为 False
    # 开启后只输出检索中间产物，不调用 LLM 生成回答
    parser.add_argument(
        "--debug", action="store_true",
        help="检索诊断模式：只输出检索结果和中间产物，不调用 LLM 生成回答"
    )

    args = parser.parse_args()

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 2：获取 QAService 单例
    # ════════════════════════════════════════════════════════════════════════
    # get_qa_service() 内部完成：
    #   - 加载场景配置
    #   - 初始化 FAQ/Doc 检索 Store
    #   - 装配 pipeline（路由→意图→改写→计划→变体→检索→上下文构建→LLM 生成）
    # 单例保证多个请求共享同一套初始化开销
    service = get_qa_service()

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 3：根据模式执行对应的 QAService 方法
    # ════════════════════════════════════════════════════════════════════════
    if args.debug:
        # ── 检索诊断模式 ──
        # debug_retrieval() 返回一个 dict，包含：
        #   - query: 原始问题
        #   - intent: 意图分类结果
        #   - rewritten_query: 改写后的问题
        #   - retrieval_plan: 检索计划参数
        #   - query_variants: 查询变体列表
        #   - faq_sources: FAQ 检索召回结果
        #   - doc_sources: Doc 检索召回结果
        #   - source_filter: 实际使用的 source 过滤
        # 不包含 LLM 生成的最终回答
        payload = service.debug_retrieval(
            args.query,
            source_filter=args.source,
            session_id=args.session_id,
            kb_version=args.kb_version,
        )
    else:
        # ── 流式问答模式 ──
        # stream_query() 返回一个生成器（generator），逐个产出 SSE 风格事件：
        #   {"type": "start"}       — 请求开始，含 trace_id
        #   {"type": "status", ...} — 阶段状态更新（如 "reranking", "generating"）
        #   {"type": "token", ...}  — LLM 逐 token 输出
        #   {"type": "end", ...}    — 请求结束，含检索诊断摘要和耗时统计
        # list() 将生成器的所有事件收集为列表，便于一次性 JSON 序列化输出
        payload = list(
            service.stream_query(
                args.query,
                source_filter=args.source,
                session_id=args.session_id,
                kb_version=args.kb_version,
            )
        )

    # ════════════════════════════════════════════════════════════════════════
    # 步骤 4：格式化输出
    # ════════════════════════════════════════════════════════════════════════
    # ensure_ascii=False — 中文字符不会转义为 \uXXXX，保持可读性
    # indent=2 — 每个嵌套层级缩进 2 空格
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    # 当脚本直接运行时（python demo_qa_service.py），__name__ == "__main__"
    # 当作为模块导入时（import demo_qa_service），__name__ == "demo_qa_service"
    # — 此时 main() 不被调用
    main()
