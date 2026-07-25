"""Prompt 模板选择器：根据意图、问题类别和业务场景选择最终回答模板。

这个模块只做配置分发，不查数据库、不调用模型。这样 Prompt 策略可以独立调整，
不会把模板选择逻辑散落到 RAG 主链路中。
"""

from __future__ import annotations
from qa_core.intent.question_category import infer_question_category
from qa_core.prompts.profiles import CATEGORY_PROMPT_PROFILES, DEFAULT_PROMPT_PROFILE, PROMPT_PROFILES, PromptProfile
from qa_core.scenarios.registry import ScenarioDefinition

def _scenario_prompt_context(scenario: ScenarioDefinition) -> dict[str, str]:
    """把场景配置转换成 Prompt 模板需要的变量。

    参数：
        scenario: 当前业务场景。

    返回：
        包含 assistant_name、business_domain、industry、support_contact、phone 的字典。

    调用顺序：回答准备阶段 -> _scenario_prompt_context()。
    """
    # 从场景配置中提取 prompt 模板需要的插值变量，所有 system prompt 中的 {assistant_name} 等占位符都依赖此字典填充
    return {
        "assistant_name": scenario.assistant_name,
        "business_domain": scenario.business_domain,
        "industry": scenario.industry,
        "support_contact": scenario.support_contact,
        "phone": scenario.support_contact,
    }


def build_answer_prompt_profile(
    intent: str,
    scenario: ScenarioDefinition,
    query: str,
) -> PromptProfile:
    """根据意图和问题类别选择最终回答模板。

    选择优先级（命中即返回）：
      1. 风险类问题模板：费用、合规、故障、总结等类别优先使用专用模板。
      2. 意图专属模板：FAQ_QUERY、KNOWLEDGE_QUERY、GREETING 等。
      3. 默认模板：前两者都没有命中时使用通用回答模板。

    选中模板后，会把当前场景的助手名称、业务域、行业和联系方式填入 system_template。

    参数：
        intent: 意图识别结果，例如 FAQ_QUERY、GREETING。
        scenario: 当前业务场景，用于注入模板变量。
        query: 用户原始问题，用于判断风险类别。

    返回：
        已完成场景变量填充的 PromptProfile。

    调用顺序：回答准备阶段 -> build_answer_prompt_profile()。
    """
    # 判断 RAG 回答风险类别（费用/合规/排障/总结等），优先于通用意图选择专用模板
    question_category = infer_question_category(query)
    # 三级回退策略：风险类别专用模板 → 意图专属模板 → 默认通用模板
    # CATEGORY_PROMPT_PROFILES 命中时优先于 PROMPT_PROFILES，确保费用/合规等强口径问题使用保守模板
    profile = CATEGORY_PROMPT_PROFILES.get(question_category) or PROMPT_PROFILES.get(intent, DEFAULT_PROMPT_PROFILE)
    # 注入场景上下文变量到 system_template（{assistant_name} 等占位符），模板本身只做最简替换
    context = _scenario_prompt_context(scenario)
    return PromptProfile(
        name=profile.name,
        system_template=profile.system_template.format(**context),
        user_template=profile.user_template,
        reason=profile.reason,
    )
