"""Lightweight source normalization helpers used before FAQ ingestion."""

from __future__ import annotations

from qa_core.scenarios.registry import ScenarioDefinition


def normalize_faq_source(
    subject: str,
    *,
    scenario: ScenarioDefinition,
    question: str = "",
) -> str:
    """Normalize a FAQ category to a valid scenario source.

    This helper intentionally has no pandas, Milvus, or vector-store imports so
    tests and validation code can reuse source normalization without loading the
    full ingestion stack.

    调用顺序：入库脚本或索引服务 -> normalize_faq_source()。
    """

    normalized = subject.strip().lower()
    # 先尝试精准匹配：如果分类名完全等于 valid_sources 中的某个值，直接返回，无需正则遍历
    if normalized in scenario.valid_sources:
        return normalized

    # 模糊匹配：遍历场景配置的正则模式，subject（FAQ 分类名）或 question（问题内容）
    # 任意一个命中即认为该 FAQ 属于该 source
    for source, pattern in scenario.compiled_source_patterns().items():
        if pattern.search(subject) or pattern.search(question):
            return source

    # 无法映射到任何 source：说明此 FAQ 分类不在场景配置范围内，需要先确认场景配置是否有遗漏
    raise ValueError(
        f"FAQ 分类无法映射到场景 {scenario.scenario_id} 的 valid_sources："
        f"subject={subject!r}, question={question!r}"
    )
