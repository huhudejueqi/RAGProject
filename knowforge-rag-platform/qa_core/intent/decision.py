"""Enterprise intent-decision gateway for retrieval intents.

The gateway fuses deterministic rules and the trainable intent model into one
governed decision payload. Direct/safety intents still close at the routing
layer; retrieval intents get a rule candidate, a model candidate, a final score,
candidate evidence, risk tags, and a policy version for Trace replay.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from functools import lru_cache
from typing import Any

from langchain_core.messages import BaseMessage

from qa_core.config.rules import get_rule_config
from qa_core.intent.model_classifier import BertIntentModelService, IntentModelPrediction
from qa_core.intent.question_category import infer_question_category
from qa_core.scenarios.registry import ScenarioDefinition


RETRIEVAL_INTENTS = {"FAQ_QUERY", "KNOWLEDGE_QUERY", "FOLLOW_UP"}


@dataclass(frozen=True)
class IntentDecisionPolicy:
    """Fixed V1 policy for rule/model intent arbitration."""

    policy_version: str = "intent-policy-v1-bert"
    model_min_score: float = 0.55
    agreement_score_boost: float = 0.03
    conflict_final_score: float = 0.68


POLICY = IntentDecisionPolicy()


def apply_intent_decision_gateway(
    query: str,
    history: list[BaseMessage],
    scenario: ScenarioDefinition,
    rule_result: Any,
) -> Any:
    """Fuse rule/model signals and return the governed intent result.

    The return type is intentionally the same dataclass instance type as
    ``rule_result``. This keeps existing pipeline code unchanged while exposing
    enterprise-grade diagnostics to ``IntentResult.as_dict()``.

    网关仲裁流程：
    1. 非检索类意图（直答/越界/问候）→ 直接返回规则结果，不调用模型
    2. 检索类意图（FAQ_QUERY/KNOWLEDGE_QUERY/FOLLOW_UP）→
       a. 调用 BERT 模型获取预测
       b. 融合规则分和模型分：
          - 无历史+模型判追问 → 规则保持（模型误判）
          - 模型低置信度 → 规则保持
          - 两者一致 → 取高分并加成
          - 规则为默认知识查询 → 采纳模型
          - 两者冲突 → 规则保持但降分
    """

    # 非检索类意图（直答/越界/问候）：不需要模型参与，直接返回规则结果
    if rule_result.intent not in RETRIEVAL_INTENTS:
        return _with_decision_fields(
            rule_result,
            final_intent=rule_result.intent,
            final_score=rule_result.rule_score,
            decision_policy="deterministic_route",
            risk_tags=_risk_tags(query, rule_result, scenario, final_intent=rule_result.intent),
            model_prediction=None,
            candidates=_candidate_payloads(rule_result, None),
        )

    # 检索类意图：加载 BERT 模型并获取预测
    model_prediction = _default_model().predict(query, has_history=bool(history))
    # 融合规则和模型信号，产出最终意图、置信度和决策策略标签
    final_intent, final_score, decision_policy = _fuse_decision(
        rule_result,
        model_prediction,
        has_history=bool(history),
    )
    # 如果最终意图与规则一致，保持规则原因；否则标记为模型辅助
    reason = rule_result.reason if final_intent == rule_result.intent else f"{rule_result.reason}_model_assisted"

    return _with_decision_fields(
        rule_result,
        final_intent=final_intent,
        final_score=final_score,
        decision_policy=decision_policy,
        risk_tags=_risk_tags(query, rule_result, scenario, final_intent=final_intent),
        model_prediction=model_prediction,
        candidates=_candidate_payloads(rule_result, model_prediction),
        reason=reason,
    )


def _fuse_decision(
    rule_result: Any,
    model_prediction: IntentModelPrediction,
    *,
    has_history: bool,
) -> tuple[str, float, str]:
    """融合规则分和模型分，按优先级决策树输出最终意图和置信度。

    决策优先级（从高到低）：
    1. 模型判追问但无历史 → 屏蔽模型，保持规则结果
       （BERT 容易把短句误判为追问，无历史上下文时追问没有意义）
    2. 模型置信度低于阈值(0.55) → 屏蔽模型，信任规则
    3. 规则和模型一致 → 两者互相印证，取高分并加成(+0.03)
    4. 规则为默认兜底(default_knowledge) → 采纳模型，规则本身不可靠
    5. 规则和模型冲突 → 保持规则但降分到 0.68，保守处理
    """
    # 优先级1：模型判追问但当前对话无历史 → 模型误判，保持规则
    if model_prediction.intent == "FOLLOW_UP" and not has_history:
        return rule_result.intent, rule_result.rule_score, "model_follow_up_without_history_guarded"

    # 优先级2：模型置信度不足 → 不信任模型，保持规则
    if model_prediction.score < POLICY.model_min_score:
        return rule_result.intent, rule_result.rule_score, "model_low_confidence_rule_kept"

    # 优先级3：规则和模型一致 → 互相印证，取两者高分并小幅加成
    if model_prediction.intent == rule_result.intent:
        final_score = min(
            1.0,
            max(rule_result.rule_score, model_prediction.score) + POLICY.agreement_score_boost,
        )
        return rule_result.intent, final_score, "rule_model_agreed"

    # 优先级4：规则是 default_knowledge（兜底），本身不可靠 → 采纳模型
    if rule_result.reason == "default_knowledge":
        return model_prediction.intent, model_prediction.score, "model_assisted_default"

    # 优先级5：规则和模型冲突且都不是兜底 → 保守策略，保持规则但降分
    return rule_result.intent, min(rule_result.rule_score, POLICY.conflict_final_score), "rule_model_conflict_guarded"


def _with_decision_fields(
    rule_result: Any,
    *,
    final_intent: str,
    final_score: float,
    decision_policy: str,
    risk_tags: tuple[str, ...],
    candidates: tuple[dict[str, str | float], ...],
    model_prediction: IntentModelPrediction | None,
    reason: str | None = None,
) -> Any:
    return replace(
        rule_result,
        intent=final_intent,
        reason=reason or rule_result.reason,
        requires_rewrite=rule_result.requires_rewrite or final_intent == "FOLLOW_UP",
        final_score=final_score,
        risk_tags=risk_tags,
        decision_policy=decision_policy,
        candidate_intents=candidates,
        model_score=model_prediction.score if model_prediction else None,
        model_version=model_prediction.model_version if model_prediction else None,
        policy_version=POLICY.policy_version,
    )


def _candidate_payloads(
    rule_result: Any,
    model_prediction: IntentModelPrediction | None,
) -> tuple[dict[str, str | float], ...]:
    """构建候选意图列表：规则候选始终排在第一位，模型候选随后追加。

    格式：[{intent, score, source:"rule"|"model", reason}, ...]
    用于 trace 回放时展示完整的决策依据链。
    """
    # 规则候选始终存在，排在第一位
    candidates: list[dict[str, str | float]] = [
        {
            "intent": str(rule_result.intent),
            "score": round(float(rule_result.rule_score), 4),
            "source": "rule",
            "reason": str(rule_result.reason),
        }
    ]
    # 有模型预测时，追加所有模型候选（3 个意图的分数）
    if model_prediction:
        for intent, score in model_prediction.scores.items():
            candidates.append(
                {
                    "intent": intent,
                    "score": round(float(score), 4),
                    "source": "model",
                    "reason": model_prediction.reason,
                }
            )
    return tuple(candidates)


def _risk_tags(
    query: str,
    rule_result: Any,
    scenario: ScenarioDefinition,
    *,
    final_intent: str,
) -> tuple[str, ...]:
    """生成风险标签列表，供 trace 和监控系统分类追踪。

    标签维度：
    - domain: 业务场景（如 enterprise_knowledge）
    - source: 推断的资料分类（规则推断出的）
    - risk: 问题风险类别（pricing/compliance/troubleshooting/summary）
    - context: 是否需要对话历史（追问类意图）
    - confidence: 规则置信度是否偏低（低规则分警告）
    """
    tags = [f"domain:{scenario.scenario_id}"]
    # 规则推断出了 source 分类时记录
    if rule_result.suggested_source:
        tags.append(f"source:{rule_result.suggested_source}")
    # 推断问题风险类别（费用/合规/排障/总结），非 default 时记录
    category = infer_question_category(query)
    if category != "default":
        tags.append(f"risk:{category}")
    # 追问意图标记需要对话历史
    if final_intent == "FOLLOW_UP":
        tags.append("context:history_required")
    # 规则分低于阈值时发出低置信度警告
    if rule_result.rule_score < get_rule_config().retrieval_strategy.low_rule_score_threshold:
        tags.append("confidence:low_rule_score")
    return tuple(tags)


@lru_cache(maxsize=1)
def _default_model() -> BertIntentModelService:
    return BertIntentModelService.from_settings()


def warmup_intent_decision_gateway() -> dict[str, object]:
    """Load the BERT intent model in the API process and run one sample prediction."""

    model = _default_model()
    prediction = model.predict("新人入职流程有哪些", has_history=False)
    return {
        "model_version": model.model_version,
        "labels": list(model.labels),
        "sample_intent": prediction.intent,
        "sample_score": round(prediction.score, 4),
        "policy_version": POLICY.policy_version,
    }
