"""检索查询扩展工具：为同一检索意图生成少量同义检索表达（如"Webhook" → "回调"），不改变问题含义。
"""

from __future__ import annotations
import re
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from qa_core.config.logging_config import get_logger
from qa_core.config.rules import QueryVariantReplacementRule, get_rule_config
from qa_core.prompts.constants import QUERY_VARIANT_SYSTEM_PROMPT
from qa_core.config.settings import get_settings
from qa_core.llm.client import get_chat_model
logger = get_logger(__name__)

FOLLOW_UP_REWRITE_MARKERS = ("追问：", "追问:")

class QueryVariants(BaseModel):
    """LLM 输出检索表达时使用的 Pydantic 结构化模型，避免模型输出解释性文本。

    调用顺序：QAService/RAG 管线 -> QueryVariants。
    """

    queries: list[str] = Field(default_factory=list, description="等价检索表达")


def generate_query_variants(query: str, *, enabled: bool, allow_short_structured: bool = False) -> list[str]:
    """为同一检索意图生成少量同义表达（如"流程"→"SOP"），提升召回而不改变问题含义。

    调用顺序：QAService/RAG 管线 -> generate_query_variants()。
    """
    # 加载应用全局设置（retrieval_variant_max 等检索配置）
    settings = get_settings()
    cleaned = query.strip()
    # --- 场景 1：功能关闭或无可变体空间 → 只返回原问题 ---
    # enabled=False（build_retrieval_plan 指定不启用）或 retrieval_variant_max ≤ 0（配置设为 0）时跳过。
    if not enabled or not cleaned or settings.retrieval_variant_max <= 0:
        return [cleaned]

    # --- 场景 2：短结构化问题（≤24 字 + 命中 marker）→ 不扩展 ---
    # 这种问题本身已是标准业务表述（如"入职需要哪些材料"），同义改写收益极低。
    # 追问例外："那审批呢"太短，不加 LLM 扩展就没有任何同义表达。
    if (
        _looks_like_short_structured_question(cleaned)
        and not allow_short_structured
        and not _is_rewritten_follow_up_query(cleaned)
    ):
        return [cleaned]

    # --- 场景 3：规则同义替换（零成本，不走 LLM）---
    # 从 rules.toml 里读取高频替换对，如 "流程"↔"SOP"。
    # 规则命中至少一个变体就返回，不再调 LLM。
    heuristic_variants = _heuristic_variants(cleaned, settings.retrieval_variant_max)
    if len(heuristic_variants) > 1:
        return heuristic_variants

    variants = [cleaned]
    # --- 场景 4：规则未覆盖 → LLM 扩展（兜底）---
    # 规则替换没命中（如"头疼"不在替换表里），回退到 LLM 生成同义变体。
    # 用纯文本 prompt + 手动 JSON 解析，兼容不支持 response_format 的模型。
    model = get_chat_model(streaming=False)
    try:
        result = model.invoke(
            [
                SystemMessage(content=QUERY_VARIANT_SYSTEM_PROMPT),
                HumanMessage(content=f"原问题：{cleaned}\n最多生成 {settings.retrieval_variant_max} 条检索表达。\n请只返回 JSON 数组，不要其他内容。"),
            ]
        )
        import json
        raw = result.content.strip()
        # 处理模型可能用 ```json ... ``` 包裹的情况
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("\n", 1)[0] if "\n" in raw else raw[3:-3]
        parsed = json.loads(raw)
        queries = parsed if isinstance(parsed, list) else parsed.get("queries", [])
        for item in queries:
            candidate = str(item).strip()
            if candidate and candidate not in variants:
                variants.append(candidate)
            if len(variants) >= settings.retrieval_variant_max + 1:
                break
    except Exception:
        logger.warning("LLM 检索变体生成失败，仅使用原问题检索: %s", cleaned)
    return variants


def _heuristic_variants(query: str, max_extra: int) -> list[str]:
    """用配置中的确定性规则为高频业务知识说法生成同义变体。

    调用顺序：QAService/RAG 管线 -> _heuristic_variants()。
    """
    variants = [query]
    rules = get_rule_config().query_variants

    def add(candidate: str) -> None:
        """在保持顺序和上限的前提下，追加非空不重复变体。

        调用顺序：QAService/RAG 管线 -> add()。
        """
        candidate = candidate.strip()
        if candidate and candidate not in variants and len(variants) < max_extra + 1:
            variants.append(candidate)

    for rule in rules.replacements:
        if not rule.matches(query):
            continue
        for old, new in rule.replacements:
            add(_replace_term(query, old, new, rule))
    return variants


def _looks_like_short_structured_question(query: str) -> bool:
    """判断问题的常见同义说法是否已被配置规则覆盖，无需进一步 LLM 扩展。

    调用顺序：QAService/RAG 管线 -> _looks_like_short_structured_question()。
    """
    return get_rule_config().query_variants.is_short_structured_question(query)


def _is_rewritten_follow_up_query(query: str) -> bool:
    """判断是否为追问改写产物，例如"报销流程是什么；追问：那审批呢"。

    调用顺序：QAService/RAG 管线 -> _is_rewritten_follow_up_query()。
    """
    return any(marker in query for marker in FOLLOW_UP_REWRITE_MARKERS)


def _replace_term(query: str, old: str, new: str, rule: QueryVariantReplacementRule) -> str:
    """Apply one configured replacement, optionally case-insensitive.

    调用顺序：QAService/RAG 管线 -> _replace_term()。
    """

    if not rule.ignore_case:
        return query.replace(old, new)
    return re.sub(re.escape(old), new, query, flags=re.IGNORECASE)
