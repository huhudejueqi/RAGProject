"""最终答案置信度计算。

检索命中的 ``score`` 只表示候选内容和查询的相关性排序，不等价于最终答案可信度。
本模块把检索分、上下文数量、来源数量、意图决策分和追问改写信号合并为
``answer_confidence``，供前端、Trace 和质量分析统一展示。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.documents import Document


LEVEL_LABELS = {
    "high": "高",
    "medium": "中",
    "low": "低",
}


@dataclass(frozen=True)
class AnswerConfidence:
    """最终答案置信度结果。

    score 是 [0, 1] 区间的工程评分；level 是便于前端展示的粗粒度等级。

    调用顺序：RAG 管线 -> calculate_answer_confidence() -> AnswerConfidence。
    """

    score: float
    level: str
    reasons: list[str]
    signals: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """转换为 API / Trace 可直接序列化的结构。

        调用顺序：RAG 管线 -> AnswerConfidence.as_dict()。
        """
        return {
            "score": round(self.score, 2),
            "level": self.level,
            "label": LEVEL_LABELS[self.level],
            "reasons": list(self.reasons),
            "signals": dict(self.signals),
        }


def calculate_answer_confidence(
    *,
    hit_type: str,
    retrieval_top_score: float,
    context_count: int,
    source_count: int,
    intent_rule_score: float,
    query: str,
    raw_query: str,
    rewritten_query: str | None,
    deterministic_route: bool = False,
    faq_exact_match: bool = False,
    intent_rule_candidate_score: float | None = None,
) -> AnswerConfidence:
    """计算最终答案置信度。★★★ 核心

    执行流程：
    1. 将检索排序分归一化为 [0, 1] 的相关性信号。
    2. 按分支计算基础分：确定性直答、FAQ 直出、RAG 生成、信息不足。
    3. 叠加上下文数量、来源数量、意图决策分等正向信号。
    4. 追问改写和低意图决策分只作为稳定性折扣，不参与检索排序。
    5. 输出 score、level、reasons 和 signals，便于前端展示和 Trace 复盘。

    调用顺序：RAG 管线 -> calculate_answer_confidence()。
    """
    normalized_score = normalize_retrieval_score(retrieval_top_score)
    decision_score = _clamp(intent_rule_score)
    rule_candidate_score = _clamp(intent_rule_candidate_score if intent_rule_candidate_score is not None else intent_rule_score)
    reasons: list[str] = []

    if deterministic_route:
        score = 0.78 + 0.18 * decision_score
        reasons.append("deterministic_route")
    elif hit_type == "faq_direct":
        if faq_exact_match:
            score = 0.95
            reasons.append("faq_exact_match")
        else:
            score = 0.55 + 0.35 * normalized_score + 0.08 * decision_score
            reasons.append("faq_score_direct")
    elif hit_type == "insufficient_context":
        score = 0.12 + 0.15 * normalized_score
        reasons.append("insufficient_context")
    else:
        score = (
            0.20
            + 0.45 * normalized_score
            + min(context_count, 4) * 0.06
            + min(source_count, 3) * 0.04
            + 0.12 * decision_score
        )
        reasons.append("rag_with_context" if context_count else "rag_without_context")

    history_rewrite_used = _history_rewrite_used(
        query=query,
        raw_query=raw_query,
        rewritten_query=rewritten_query,
    )
    if history_rewrite_used:
        score -= 0.05 if context_count >= 2 else 0.08
        reasons.append("history_rewrite_used")

    if decision_score < 0.70 and not deterministic_route:
        score -= 0.04
        reasons.append("low_intent_decision_score")

    if hit_type == "rag" and context_count == 0:
        score = min(score, 0.35)
        reasons.append("no_selected_context")

    final_score = round(_clamp(score), 2)
    return AnswerConfidence(
        score=final_score,
        level=confidence_level(final_score),
        reasons=reasons,
        signals={
            "hit_type": hit_type,
            "retrieval_top_score": retrieval_top_score,
            "normalized_retrieval_score": round(normalized_score, 2),
            "context_count": context_count,
            "source_count": source_count,
            "intent_rule_score": round(rule_candidate_score, 2),
            "intent_decision_score": round(decision_score, 2),
            "history_rewrite_used": history_rewrite_used,
            "faq_exact_match": faq_exact_match,
            "deterministic_route": deterministic_route,
        },
    )


def record_answer_confidence(
    context: Any,
    *,
    hit_type: str,
    retrieval_top_score: float,
    context_count: int,
    source_count: int,
    deterministic_route: bool = False,
    faq_exact_match: bool = False,
) -> dict[str, Any]:
    """计算并写入请求上下文，作为最终 end 事件和 Trace 的统一字段。

    调用顺序：RAG 管线 -> record_answer_confidence()。
    """
    confidence = calculate_answer_confidence(
        hit_type=hit_type,
        retrieval_top_score=retrieval_top_score,
        context_count=context_count,
        source_count=source_count,
        intent_rule_score=float(context.intent_payload["confidence"]) if context.intent_payload else 0.6,
        intent_rule_candidate_score=float(context.intent_payload["rule_score"]) if context.intent_payload else 0.6,
        query=context.query,
        raw_query=context.raw_query,
        rewritten_query=context.rewritten_query,
        deterministic_route=deterministic_route,
        faq_exact_match=faq_exact_match,
    ).as_dict()
    context.answer_confidence = confidence
    context.retrieval_info["answer_confidence"] = confidence
    return confidence


def faq_exact_match(query: str, doc: Document | None) -> bool:
    """判断 FAQ 命中是否为标准问题精确匹配。

    调用顺序：RAG 管线 -> faq_exact_match()。
    """
    if doc is None:
        return False
    metadata = doc.metadata
    standard_question = str(metadata.get("standard_question") or metadata.get("question") or doc.page_content).strip()
    return query.strip() == standard_question


def normalize_retrieval_score(score: float) -> float:
    """把检索/重排分数压到 [0, 1]，仅用于置信度派生，不改变排序。

    Milvus/LangChain 返回值在本项目里按“越大越相关”使用；CrossEncoder 有时会返回
    大于 1 的 logits，因此这里用平滑压缩而不是直接截断。

    调用顺序：RAG 管线 -> normalize_retrieval_score()。
    """
    if score <= 0:
        return 0.0
    if score <= 1:
        return score
    return 1 - (1 / (1 + score))


def confidence_level(score: float) -> str:
    """将连续分数映射为前端展示等级。

    调用顺序：RAG 管线 -> confidence_level()。
    """
    if score >= 0.82:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def _history_rewrite_used(*, query: str, raw_query: str, rewritten_query: str | None) -> bool:
    """判断当前回答是否依赖历史追问改写。

    调用顺序：RAG 管线 -> _history_rewrite_used()。
    """
    rewritten = (rewritten_query or "").strip()
    if not rewritten:
        return False
    return rewritten not in {query.strip(), raw_query.strip()}


def _clamp(value: float) -> float:
    """将数值限制在 [0, 1] 区间。

    调用顺序：RAG 管线 -> _clamp()。
    """
    return max(0.0, min(1.0, value))
